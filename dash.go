package main

import (
	_ "embed"
	"encoding/json"
	"net/http"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/openfluke/welvet/lucy"
)

//go:embed index.html
var dashHTML []byte

//go:embed report.html
var reportHTML []byte

// leafRow is one variant inside the active tree (LR × cam × grid).
type leafRow struct {
	ID      string  `json:"id"`
	LR      float64 `json:"lr"`
	Cams    int     `json:"cams"`
	GridN   int     `json:"grid_n"`
	Phase   string  `json:"phase,omitempty"`
	Acc     float64 `json:"acc"`
	Soft    float64 `json:"soft_acc"`
	Avail   float64 `json:"availability"`
	Thru    float64 `json:"throughput"`
	Score   float64 `json:"score"`
	RAMKiB  float64 `json:"ram_kib"`
	Levels  int     `json:"levels"`
	AccΔ    float64 `json:"acc_delta,omitempty"`
	Improve float64 `json:"improve_pct,omitempty"`
	Done    bool    `json:"done"`
	Err     string  `json:"error,omitempty"`
}

// treeReport is appended when a whole architecture tree finishes; board clears after.
type treeReport struct {
	Index     int       `json:"index"`
	URL       string    `json:"url"`
	PDF       string    `json:"pdf"`
	Key       string    `json:"key"`
	Mode      string    `json:"mode"`
	Layer     string    `json:"layer"`
	DType     string    `json:"dtype"`
	Challenge string    `json:"challenge"`
	Leaves    int       `json:"leaves"`
	BestID    string    `json:"best_id"`
	BestAcc   float64   `json:"best_acc"`
	BestScore float64   `json:"best_score"`
	BestΔ     float64   `json:"best_acc_delta"`
	BestLR    float64   `json:"best_lr"`
	BestCams  int       `json:"best_cams"`
	BestGrid  int       `json:"best_grid"`
	Finished  time.Time `json:"finished"`
	Rows      []leafRow `json:"rows,omitempty"`
}

type treeMeta struct {
	Key       string `json:"key"`
	Mode      string `json:"mode"`
	Layer     string `json:"layer"`
	DType     string `json:"dtype"`
	Challenge string `json:"challenge"`
	Index     int    `json:"index"` // 1-based
	Total     int    `json:"total"`
	LeafDone  int    `json:"leaf_done"`
	LeafTotal int    `json:"leaf_total"`
}

type liveHub struct {
	mu sync.RWMutex

	frame    Frame
	play     playView
	action   Action
	mode     string
	thinkK   int
	status   string
	jobID    string
	phase    string
	jobIndex int
	jobTotal int

	tree     treeMeta
	board    []leafRow // ONLY active tree — cleared on finishTree
	reports   []treeReport
	lpdByChal map[string]lucy.LPD // one LPD board per challenge (never cross-game)
	tide      *tideBridge         // optional Tide Lucy dash (second port)
	plan      sweepPlan
	started   bool
	startCh   chan struct{}
}

// sweepPlan is what Start will run (shown on the dash before the gate opens).
type sweepPlan struct {
	Label      string   `json:"label"`
	Modes      []string `json:"modes"`
	Layers     []string `json:"layers"`
	DTypes     string   `json:"dtypes"`
	Challenges string   `json:"challenges"`
	LRs        string   `json:"lrs"`
	Cams       string   `json:"cams"`
	Grids      string   `json:"grids"`
	Trees      int      `json:"trees"`
	Leaves     int      `json:"leaves"`
	Pending    int      `json:"pending"`
	Full       bool     `json:"full"`
}

func newLiveHub() *liveHub {
	return &liveHub{
		startCh: make(chan struct{}),
		status:  "waiting for Start",
		board:   nil,
		reports: nil,
	}
}

func (h *liveHub) awaitStart() { <-h.startCh }

func (h *liveHub) signalStart() {
	h.mu.Lock()
	var tide *tideBridge
	if !h.started {
		h.started = true
		h.status = "running"
		close(h.startCh)
	}
	tide = h.tide
	h.mu.Unlock()
	if tide != nil {
		tide.signalStart()
	}
}

func (h *liveHub) setStatus(s string) {
	h.mu.Lock()
	h.status = s
	h.mu.Unlock()
}

func (h *liveHub) setMeta(jobID, phase string, index, total int) {
	h.mu.Lock()
	h.jobID = jobID
	h.phase = phase
	h.jobIndex = index
	h.jobTotal = total
	h.mu.Unlock()
}

func (h *liveHub) setFrame(fr Frame, act Action, mode string, thinkK int) {
	h.mu.Lock()
	h.frame = fr
	h.play = extractPlay(fr)
	h.action = act
	h.mode = mode
	h.thinkK = thinkK
	h.mu.Unlock()
}

// playView is normalized 0..1 agent/goal positions for the dash canvas.
type playView struct {
	AgentX float64 `json:"agent_x"`
	AgentY float64 `json:"agent_y"`
	GoalX  float64 `json:"goal_x"`
	GoalY  float64 `json:"goal_y"`
	OK     bool    `json:"ok"`
}

// extractPlay finds agent (color 1) and goal (color 2) in frame.layers[0].
func extractPlay(fr Frame) playView {
	pv := playView{AgentX: 0.5, AgentY: 0.5, GoalX: 0.8, GoalY: 0.2}
	if len(fr.Layers) == 0 || len(fr.Layers[0]) == 0 {
		return pv
	}
	layer := fr.Layers[0]
	h := len(layer)
	w := len(layer[0])
	if h < 1 || w < 1 {
		return pv
	}
	foundA, foundG := false, false
	for y := 0; y < h; y++ {
		row := layer[y]
		for x := 0; x < len(row) && x < w; x++ {
			switch row[x] {
			case 1: // agent
				pv.AgentX = (float64(x) + 0.5) / float64(w)
				pv.AgentY = (float64(y) + 0.5) / float64(h)
				foundA = true
			case 2: // goal
				pv.GoalX = (float64(x) + 0.5) / float64(w)
				pv.GoalY = (float64(y) + 0.5) / float64(h)
				foundG = true
			}
		}
	}
	pv.OK = foundA || foundG
	return pv
}

func (h *liveHub) setLPDs(by map[string]lucy.LPD) {
	h.mu.Lock()
	h.lpdByChal = by
	h.mu.Unlock()
}

// beginTree resets the job board to this tree's empty leaf slots.
func (h *liveHub) beginTree(t Tree, treeIdx, treeTotal int) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.tree = treeMeta{
		Key: t.Key, Mode: t.Mode, Layer: t.Layer, DType: t.DType, Challenge: t.Challenge,
		Index: treeIdx, Total: treeTotal, LeafDone: 0, LeafTotal: len(t.Jobs),
	}
	h.board = make([]leafRow, len(t.Jobs))
	for i, j := range t.Jobs {
		h.board[i] = leafRow{
			ID: j.ID, LR: j.LR, Cams: j.Cams, GridN: j.GridN, Phase: "pending",
		}
	}
	h.status = fmtTreeStatus(h.tree, "start")
}

func fmtTreeStatus(t treeMeta, verb string) string {
	return "tree " + itoa(t.Index) + "/" + itoa(t.Total) + " " + verb + " · " +
		t.Mode + " · " + t.Layer + "/" + t.DType + " · " + t.Challenge
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	var b [16]byte
	i := len(b)
	for n > 0 {
		i--
		b[i] = byte('0' + n%10)
		n /= 10
	}
	return string(b[i:])
}

// pulseLeaf updates one leaf in the active tree board (does not grow forever).
func (h *liveHub) pulseLeaf(r modeResult) {
	h.mu.Lock()
	defer h.mu.Unlock()
	id := r.ID
	if id == "" {
		id = r.Mode
	}
	// strip phase suffix for matching tree leaf ids
	base := id
	for _, suf := range []string{"|after_train", "|after_freeze", "|freeze", "|after"} {
		if len(base) > len(suf) && base[len(base)-len(suf):] == suf {
			base = base[:len(base)-len(suf)]
			break
		}
	}
	for i := range h.board {
		if h.board[i].ID != base && h.board[i].ID != id {
			continue
		}
		h.board[i].Phase = r.Phase
		h.board[i].Acc = r.Acc
		h.board[i].Soft = r.Soft
		h.board[i].Avail = r.Avail
		h.board[i].Thru = r.Thru
		h.board[i].Score = r.Score
		h.board[i].RAMKiB = r.RAMKiB
		h.board[i].Levels = r.Levels
		h.board[i].AccΔ = r.AccDelta
		h.board[i].Improve = r.ImprovePct
		h.board[i].Err = r.Err
		if r.Phase == "train" || r.Phase == "after_train" || r.Phase == "" {
			if r.Err != "" || r.Phase == "after_train" || r.Phase == "train" {
				// mark done only on final train row finishMode path
			}
		}
		break
	}
}

func (h *liveHub) finishLeaf(r modeResult) {
	h.mu.Lock()
	defer h.mu.Unlock()
	id := r.ID
	for i := range h.board {
		if h.board[i].ID != id {
			continue
		}
		h.board[i].Phase = "done"
		h.board[i].Acc = r.Acc
		h.board[i].Soft = r.Soft
		h.board[i].Avail = r.Avail
		h.board[i].Thru = r.Thru
		h.board[i].Score = r.Score
		h.board[i].RAMKiB = r.RAMKiB
		h.board[i].Levels = r.Levels
		h.board[i].AccΔ = r.AccDelta
		h.board[i].Improve = r.ImprovePct
		h.board[i].Err = r.Err
		h.board[i].Done = true
		h.tree.LeafDone++
		break
	}
	h.status = fmtTreeStatus(h.tree, "leaf")
}

// finishTree appends a consolidation report and CLEARS the job board.
func (h *liveHub) finishTree(rep treeReport) {
	h.mu.Lock()
	defer h.mu.Unlock()
	rep.Index = len(h.reports) + 1
	rep.URL = "/report/" + itoa(rep.Index)
	rep.PDF = "/report/" + itoa(rep.Index) + ".pdf"
	if len(rep.Rows) == 0 && len(h.board) > 0 {
		rep.Rows = append([]leafRow(nil), h.board...)
	}
	h.reports = append(h.reports, rep)
	h.board = nil
	h.status = fmtTreeStatus(h.tree, "archived") + " → report #" + itoa(len(h.reports))
	h.tree.LeafDone = h.tree.LeafTotal
}

func (h *liveHub) reportByIndex(n int) (treeReport, bool) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	if n < 1 || n > len(h.reports) {
		return treeReport{}, false
	}
	return h.reports[n-1], true
}

// seedReports loads prior consolidation reports on resume (no board wipe).
func (h *liveHub) seedReports(reps []treeReport) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.reports = nil
	for i, rep := range reps {
		rep.Index = i + 1
		rep.URL = "/report/" + itoa(rep.Index)
		rep.PDF = "/report/" + itoa(rep.Index) + ".pdf"
		h.reports = append(h.reports, rep)
	}
	if len(h.reports) > 0 {
		if !h.started {
			h.status = "waiting for Start · " + itoa(len(h.reports)) + " report(s) ready"
		} else {
			h.status = "resumed · " + itoa(len(h.reports)) + " consolidation report(s) loaded"
		}
	}
}

// Legacy aliases used by runCfg pulse paths.
func (h *liveHub) pulseMode(r modeResult) {
	h.pulseLeaf(r)
	h.mu.RLock()
	t := h.tide
	h.mu.RUnlock()
	if t != nil {
		t.pulseRunning(r)
	}
}
func (h *liveHub) finishMode(r modeResult) { h.pulseLeaf(r) }

func (h *liveHub) setTide(t *tideBridge) {
	h.mu.Lock()
	h.tide = t
	h.mu.Unlock()
}

func (h *liveHub) setPlan(p sweepPlan) {
	h.mu.Lock()
	h.plan = p
	if !h.started {
		h.status = "waiting for Start · " + p.Label + " · " +
			itoa(p.Trees) + " trees / " + itoa(p.Leaves) + " leaves (" + itoa(p.Pending) + " pending)"
	}
	h.mu.Unlock()
}

// sortReports best → worst by tree best Score, then Acc, then Δacc.
func sortReports(reps []treeReport) {
	sort.Slice(reps, func(i, j int) bool {
		a, b := reps[i], reps[j]
		if a.BestScore != b.BestScore {
			return a.BestScore > b.BestScore
		}
		if a.BestAcc != b.BestAcc {
			return a.BestAcc > b.BestAcc
		}
		return a.BestΔ > b.BestΔ
	})
}

func (h *liveHub) snapshot() livePayload {
	h.mu.RLock()
	defer h.mu.RUnlock()
	board := append([]leafRow(nil), h.board...)
	reports := append([]treeReport(nil), h.reports...)
	sortReports(reports)
	return livePayload{
		At:       time.Now().UTC(),
		Status:   h.status,
		Mode:     h.mode,
		ThinkK:   h.thinkK,
		Action:   h.action,
		Frame:    h.frame,
		Play:     h.play,
		JobID:    h.jobID,
		Phase:    h.phase,
		JobIndex: h.jobIndex,
		JobTotal: h.jobTotal,
		Tree:     h.tree,
		Board:    board,
		Reports:  reports,
		Started:         h.started,
		LPDByChallenge:  h.lpdByChal,
		ActiveChallenge: h.tree.Challenge,
		TideURL:         tideURLFrom(h.tide),
		Plan:            h.plan,
	}
}

func tideURLFrom(t *tideBridge) string {
	if t == nil || t.srv == nil {
		return ""
	}
	return "/tide → :" + dashPort(t.srv.Addr)
}

type livePayload struct {
	At               time.Time            `json:"at"`
	Status           string               `json:"status"`
	Mode             string               `json:"mode"`
	ThinkK           int                  `json:"think_k"`
	Action           Action               `json:"action"`
	Frame            Frame                `json:"frame"`
	Play             playView             `json:"play"`
	JobID            string               `json:"job_id,omitempty"`
	Phase            string               `json:"phase,omitempty"`
	JobIndex         int                  `json:"job_index,omitempty"`
	JobTotal         int                  `json:"job_total,omitempty"`
	Tree             treeMeta             `json:"tree"`
	Board            []leafRow            `json:"board"`
	Reports          []treeReport         `json:"reports"`
	Started          bool                 `json:"started"`
	LPDByChallenge   map[string]lucy.LPD  `json:"lpd_by_challenge"`
	ActiveChallenge  string               `json:"active_challenge,omitempty"`
	TideURL          string               `json:"tide_url,omitempty"`
	Plan             sweepPlan            `json:"plan"`
}

type dashServer struct {
	addr string
	hub  *liveHub
}

func newDashServer(addr string, hub *liveHub) *dashServer {
	return &dashServer{addr: addr, hub: hub}
}

func (d *dashServer) awaitStart()  { d.hub.awaitStart() }
func (d *dashServer) signalStart() { d.hub.signalStart() }

func (d *dashServer) listen() error {
	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Header().Set("Cache-Control", "no-store")
		_, _ = w.Write(dashHTML)
	})
	mux.HandleFunc("/report/", func(w http.ResponseWriter, r *http.Request) {
		path := strings.TrimPrefix(r.URL.Path, "/report/")
		if path == "" {
			http.NotFound(w, r)
			return
		}
		wantPDF := strings.HasSuffix(path, ".pdf")
		idStr := strings.TrimSuffix(path, ".pdf")
		n, err := strconv.Atoi(idStr)
		if err != nil || n < 1 {
			http.Error(w, "bad report id", http.StatusBadRequest)
			return
		}
		rep, ok := d.hub.reportByIndex(n)
		if !ok {
			http.Error(w, "report not found", http.StatusNotFound)
			return
		}
		if wantPDF {
			pdf := buildReportPDF(rep)
			w.Header().Set("Content-Type", "application/pdf")
			w.Header().Set("Content-Disposition", `inline; filename="test51-tree-`+itoa(n)+`.pdf"`)
			w.Header().Set("Cache-Control", "no-store")
			_, _ = w.Write(pdf)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Header().Set("Cache-Control", "no-store")
		_, _ = w.Write(reportHTML)
	})
	mux.HandleFunc("/api/report/", func(w http.ResponseWriter, r *http.Request) {
		idStr := strings.TrimPrefix(r.URL.Path, "/api/report/")
		idStr = strings.TrimSuffix(idStr, "/")
		n, err := strconv.Atoi(idStr)
		if err != nil || n < 1 {
			http.Error(w, "bad report id", http.StatusBadRequest)
			return
		}
		rep, ok := d.hub.reportByIndex(n)
		if !ok {
			http.Error(w, "report not found", http.StatusNotFound)
			return
		}
		writeJSON(w, rep)
	})
	mux.HandleFunc("/api/live", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, d.hub.snapshot())
	})
	mux.HandleFunc("/api/board", func(w http.ResponseWriter, r *http.Request) {
		snap := d.hub.snapshot()
		writeJSON(w, map[string]any{
			"tree":              snap.Tree,
			"board":             snap.Board,
			"reports":           snap.Reports,
			"lpd_by_challenge":  snap.LPDByChallenge,
			"active_challenge":  snap.ActiveChallenge,
		})
	})
	mux.HandleFunc("/api/start", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", http.StatusMethodNotAllowed)
			return
		}
		d.hub.signalStart()
		writeJSON(w, map[string]any{"ok": true})
	})
	mux.HandleFunc("/api/meta", func(w http.ResponseWriter, r *http.Request) {
		tideURL := ""
		d.hub.mu.RLock()
		if d.hub.tide != nil && d.hub.tide.srv != nil {
			tideURL = "http://<host>:" + dashPort(d.hub.tide.srv.Addr)
		}
		d.hub.mu.RUnlock()
		writeJSON(w, map[string]any{
			"name": "test51",
			"apis": map[string]string{
				"live":   "/api/live",
				"board":  "/api/board",
				"start":  "POST /api/start",
				"report": "/report/{n}",
				"pdf":    "/report/{n}.pdf",
			},
			"tide_url": tideURL,
		})
	})
	srv := &http.Server{Addr: d.addr, Handler: withCORS(mux)}
	return srv.ListenAndServe()
}

func withCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(v)
}
