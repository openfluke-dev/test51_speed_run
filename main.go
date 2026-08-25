package main

import (
	"bufio"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"net"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/openfluke/welvet/architecture"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/lucy"
)

type modeResult struct {
	ID         string        `json:"id,omitempty"`
	Mode       string        `json:"mode"`
	Layer      string        `json:"layer,omitempty"`
	DType      string        `json:"dtype,omitempty"`
	LR         float64       `json:"lr,omitempty"`
	Challenge  string        `json:"challenge,omitempty"`
	Cams       int           `json:"cams,omitempty"`
	GridN      int           `json:"grid_n,omitempty"`
	Phase      string        `json:"phase,omitempty"`
	Acc        float64       `json:"acc"`
	Soft       float64       `json:"soft_acc"`
	Avail      float64       `json:"availability"`
	Thru       float64       `json:"throughput"`
	Score      float64       `json:"score"`
	RAMKiB     float64       `json:"ram_kib"`
	Levels     int           `json:"levels"`
	Actions    int64         `json:"actions"`
	ThinkK     int           `json:"think_k"`
	AccDelta   float64       `json:"acc_delta,omitempty"`
	ScoreDelta float64       `json:"score_delta,omitempty"`
	ImprovePct float64       `json:"improve_pct,omitempty"` // AccΔ relative to train Acc
	Promoted   bool          `json:"promoted,omitempty"`
	Err        string        `json:"error,omitempty"`
	Lucy       lucy.Snapshot `json:"lucy"`
	Weights    [][]float32   `json:"-"`
}

type runCfg struct {
	mode      parallel.TrainMode
	initW     [][]float32
	dur       time.Duration
	win       time.Duration
	lr        float64
	seed      int64
	thinkK    int
	game      string
	challenge string
	bridgePy  string
	layer     string
	dtype     core.DType
	cams      int
	gridN     int
	train     bool // false = freeze weights (think+act only)
	phase     string
	jobID     string
	hub       *liveHub
	promoted  bool
}


func summarizeTree(t Tree, leaves []modeResult) treeReport {
	rep := treeReport{
		Key: t.Key, Mode: t.Mode, Layer: t.Layer, DType: t.DType, Challenge: t.Challenge,
		Leaves: len(t.Jobs), Finished: time.Now().UTC(),
	}
	best := -1.0
	for _, r := range leaves {
		row := leafRow{
			ID: r.ID, LR: r.LR, Cams: r.Cams, GridN: r.GridN, Phase: r.Phase,
			Acc: r.Acc, Soft: r.Soft, Avail: r.Avail, Thru: r.Thru, Score: r.Score, RAMKiB: r.RAMKiB,
			Levels: r.Levels, AccΔ: r.AccDelta, Improve: r.ImprovePct, Done: true, Err: r.Err,
		}
		rep.Rows = append(rep.Rows, row)
		if r.Err != "" {
			continue
		}
		if r.Phase != "" && r.Phase != "train" && r.Phase != "after_train" {
			continue
		}
		score := r.Score
		if score > best || (score == best && r.Acc > rep.BestAcc) {
			best = score
			rep.BestID = r.ID
			rep.BestAcc = r.Acc
			rep.BestScore = r.Score
			rep.BestΔ = r.AccDelta
			rep.BestLR = r.LR
			rep.BestCams = r.Cams
			rep.BestGrid = r.GridN
		}
	}
	return rep
}


func baseJobID(id string) string {
	for _, suf := range []string{"|after_train", "|after_freeze", "|freeze", "|after", "|promote"} {
		if len(id) > len(suf) && id[len(id)-len(suf):] == suf {
			return id[:len(id)-len(suf)]
		}
	}
	return id
}

func phaseRank(phase string) int {
	switch phase {
	case "after_train":
		return 3
	case "train", "":
		return 2
	case "after_freeze", "freeze":
		return 1
	default:
		return 0
	}
}

// rebuildReportsFromResults rebuilds consolidation reports for fully-done trees from ckpt rows.
func rebuildReportsFromResults(trees []Tree, results []modeResult, done map[string]bool) []treeReport {
	byID := map[string]modeResult{}
	for _, r := range results {
		id := baseJobID(r.ID)
		prev, ok := byID[id]
		if !ok || phaseRank(r.Phase) >= phaseRank(prev.Phase) {
			rr := r
			rr.ID = id
			byID[id] = rr
		}
	}
	var out []treeReport
	for _, tree := range trees {
		allDone := true
		for _, j := range tree.Jobs {
			if !done[j.ID] {
				allDone = false
				break
			}
		}
		if !allDone {
			continue
		}
		leaves := make([]modeResult, 0, len(tree.Jobs))
		for _, j := range tree.Jobs {
			if r, ok := byID[j.ID]; ok {
				leaves = append(leaves, r)
			}
		}
		if len(leaves) == 0 {
			continue
		}
		out = append(out, summarizeTree(tree, leaves))
	}
	return out
}

// enrichReportsFromResults fills missing throughput on saved report rows from results.json.
// Older reports were saved before throughput was copied into leaf rows — no re-run needed.
func enrichReportsFromResults(reports []treeReport, results []modeResult) bool {
	byID := map[string]modeResult{}
	for _, r := range results {
		id := baseJobID(r.ID)
		prev, ok := byID[id]
		if !ok || phaseRank(r.Phase) >= phaseRank(prev.Phase) {
			rr := r
			rr.ID = id
			byID[id] = rr
		}
	}
	changed := false
	for ri := range reports {
		for li := range reports[ri].Rows {
			row := &reports[ri].Rows[li]
			m, ok := byID[row.ID]
			if !ok {
				continue
			}
			if row.Thru == 0 && m.Thru > 0 {
				row.Thru = m.Thru
				changed = true
			}
		}
	}
	return changed
}

func main() {
	LoadDotEnv(".env")

	modesFlag := flag.String("modes", EnvOr("SPEED_MODES", EnvOr("TEST51_MODES", "all")), "all named modes, or csv")
	thinkK := flag.Int("think", EnvInt("SPEED_THINK", 4), "think steps")
	dur := flag.Duration("duration", EnvDuration("SPEED_DURATION", 3*time.Second), "train phase wall")
	afterFreeze := flag.Duration("after-freeze", EnvDuration("SPEED_AFTER_FREEZE", 2*time.Second), "freeze phase")
	afterTrain := flag.Duration("after-train", EnvDuration("SPEED_AFTER_TRAIN", 3*time.Second), "after-train phase")
	promote := flag.Duration("promote", EnvDuration("SPEED_PROMOTE", 0), "LPD promote wall (0=skip)")
	win := flag.Duration("window", EnvDuration("SPEED_WINDOW", time.Second), "Lucy window")
	lr := flag.Float64("lr", EnvFloat("SPEED_LR", 0.05), "single LR if -lrs empty")
	lrsFlag := flag.String("lrs", EnvOr("SPEED_LRS", "funny"), "funny|csv")
	layerFlag := flag.String("layer", EnvOr("SPEED_LAYER", EnvOr("TEST51_LAYERS", "")), "dense|dense-wide|…|all|csv (empty = ask)")
	dtypesFlag := flag.String("dtypes", EnvOr("SPEED_DTYPES", "all"), "float32|all|csv")
	challengesFlag := flag.String("challenges", EnvOr("SPEED_CHALLENGES", "all"), "chase|…|all")
	camsFlag := flag.String("cams", EnvOr("SPEED_CAMS", "1-3"), "1-3|all")
	gridsFlag := flag.String("grids", EnvOr("SPEED_GRIDS", "1-3"), "1-3|all")
	full := flag.Bool("full", EnvBool("SPEED_FULL", true), "full permute around chosen layer")
	workersFlag := flag.Int("workers", EnvInt("SPEED_WORKERS", 0), "leaf workers (0 = NumCPU); ~1 core each")
	seed := flag.Int64("seed", int64(EnvInt("SPEED_SEED", 1)), "rng seed")
	addr := flag.String("addr", EnvOr("SPEED_ADDR", "0.0.0.0:5151"), "dash listen")
	tideAddr := flag.String("tide-addr", EnvOr("SPEED_TIDE_ADDR", "0.0.0.0:8080"), "Tide dash (empty=off)")
	ckptRoot := flag.String("ckpt-root", EnvOr("SPEED_CKPT_ROOT", "speed_ckpt"), "per-layer ckpt root")
	resume := flag.Bool("resume", EnvBool("SPEED_RESUME", true), "skip done job IDs")
	autoStart := flag.Bool("autostart", EnvBool("SPEED_AUTOSTART", true), "skip Start gate")
	flag.Parse()

	layers, err := resolveLayers(*layerFlag)
	must(err)
	if len(layers) == 0 {
		fmt.Fprintln(os.Stderr, "no layers selected")
		os.Exit(2)
	}

	if *full {
		*dtypesFlag = "all"
		*lrsFlag = "funny"
		*challengesFlag = "all"
		*camsFlag = "1-3"
		*gridsFlag = "1-3"
	}

	modes, err := parseModeList(*modesFlag)
	must(err)
	dtypes, err := parseDTypeList(*dtypesFlag)
	must(err)
	var lrs []float64
	if strings.TrimSpace(*lrsFlag) == "" {
		lrs = []float64{*lr}
	} else {
		lrs, err = parseLRList(*lrsFlag)
		must(err)
	}
	challenges, err := parseChallengeList(*challengesFlag)
	must(err)
	cams, err := parseCamsList(*camsFlag)
	must(err)
	grids, err := parseGridList(*gridsFlag)
	must(err)


	workers := *workersFlag
	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	if workers < 1 {
		workers = 1
	}
	leafMultiCore = workers == 1

	allJobs := expandJobs(modes, layers, dtypes, lrs, challenges, cams, grids)
	allTrees := groupTrees(allJobs)
	hub := newLiveHub()
	hub.setPlan(buildSweepPlan(modes, layers, *dtypesFlag, *challengesFlag, *lrsFlag, *camsFlag, *gridsFlag,
		len(allTrees), len(allJobs), len(allJobs), *full))

	var dash *dashServer
	if strings.TrimSpace(*addr) != "" {
		listen := normalizeDashAddr(*addr)
		dash = newDashServer(listen, hub)
		go func() {
			if err := dash.listen(); err != nil {
				fmt.Fprintf(os.Stderr, "dash: %v\n", err)
			}
		}()
		fmt.Printf("dash  http://<host>:%s\n", dashPort(listen))
	}
	tide := startTideBridge(*tideAddr, allJobs, *lr, "speed-run")
	hub.setTide(tide)
	if dash != nil {
		if !*autoStart {
			fmt.Println("waiting for Start on dash…")
			dash.awaitStart()
		} else {
			dash.signalStart()
		}
	} else if tide != nil && *autoStart {
		tide.signalStart()
	}

	fmt.Println("╔══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║  test51 speed-run · N workers ≈ 1 core each                     ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════╝")
	fmt.Printf("layers=%v  workers=%d  leafMultiCore=%v  modes=%d  totalJobs=%d\n\n",
		layers, workers, leafMultiCore, len(modes), len(allJobs))

	type leafOut struct {
		job   Job
		train modeResult
		extra []modeResult
	}

	for li, layer := range layers {
		jobs := expandJobs(modes, []string{layer}, dtypes, lrs, challenges, cams, grids)
		trees := groupTrees(jobs)
		ckpt := filepath.Join(*ckptRoot, layer)
		store := NewStore(ckpt)
		prog, _, err := store.Load()
		must(err)
		done := doneSet(prog.DoneIDs)
		if !*resume {
			done = map[string]bool{}
			prog = &Progress{}
		}
		prog.Total = len(jobs)

		pending := 0
		for _, j := range jobs {
			if !done[j.ID] {
				pending++
			}
		}
		hub.setPlan(buildSweepPlan(modes, []string{layer}, *dtypesFlag, *challengesFlag, *lrsFlag, *camsFlag, *gridsFlag,
			len(trees), len(jobs), pending, *full))
		hub.setStatus(fmt.Sprintf("layer %d/%d · %s · %d pending", li+1, len(layers), layer, pending))
		fmt.Printf("── layer %d/%d  %s  jobs=%d pending=%d  ckpt=%s ──\n",
			li+1, len(layers), layer, len(jobs), pending, ckpt)

		var results []modeResult
		if *resume && len(prog.Completed) > 0 {
			results = append(results, prog.Completed...)
		}
		disk, _ := store.LoadResults()
		if *resume && len(disk.Results) > len(results) && len(results) == 0 {
			results = append(results, disk.Results...)
		}
		if *resume {
			var seeded []treeReport
			if len(disk.Reports) > 0 {
				seeded = disk.Reports
			} else {
				seeded = rebuildReportsFromResults(trees, results, done)
			}
			if len(seeded) > 0 {
				hub.seedReports(seeded)
			}
			if tide != nil && len(results) > 0 {
				tide.seedCompleted(results)
			}
		}

		if pending == 0 {
			fmt.Printf("  (layer %s already complete — skip)\n\n", layer)
			continue
		}

		jobCh := make(chan Job, workers*2)
		outCh := make(chan leafOut, workers*2)
		var wg sync.WaitGroup
		for w := 0; w < workers; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for job := range jobCh {
					hub.setStatus(fmt.Sprintf("%s · workers=%d · %s", layer, workers, shortID(job.ID)))
					trainR := runCfg{
						mode: job.Mode, dur: *dur, win: *win, lr: job.LR, seed: *seed + int64(hashID(job.ID)),
						thinkK: *thinkK, game: "", challenge: job.Challenge, layer: job.Layer,
						dtype: job.DType, cams: job.Cams, gridN: job.GridN, train: true,
						phase: "train", jobID: job.ID, hub: nil,
					}.run()
					trainR.ID = job.ID
					trainR.Layer = job.Layer
					trainR.DType = job.DType.String()
					trainR.LR = job.LR
					trainR.Challenge = job.Challenge
					trainR.Cams = job.Cams
					trainR.GridN = job.GridN
					trainR.Phase = "train"

					var extra []modeResult
					last := trainR
					if trainR.Err == "" && *afterFreeze > 0 && len(trainR.Weights) > 0 {
						fr := runCfg{
							mode: job.Mode, initW: trainR.Weights, dur: *afterFreeze, win: *win, lr: job.LR,
							seed: *seed + int64(hashID(job.ID)) + 1, thinkK: *thinkK, challenge: job.Challenge,
							layer: job.Layer, dtype: job.DType, cams: job.Cams, gridN: job.GridN, train: false,
							phase: "after_freeze", jobID: job.ID + "|freeze", hub: nil,
						}.run()
						fr.ID = job.ID + "|after_freeze"
						fr.Layer, fr.DType, fr.LR, fr.Challenge = job.Layer, job.DType.String(), job.LR, job.Challenge
						fr.Cams, fr.GridN = job.Cams, job.GridN
						fr.Phase = "after_freeze"
						fr.AccDelta = fr.Acc - trainR.Acc
						fr.ScoreDelta = fr.Score - trainR.Score
						extra = append(extra, fr)
						last = fr
					}
					if trainR.Err == "" && *afterTrain > 0 && len(last.Weights) > 0 {
						baseW := last.Weights
						if len(trainR.Weights) > 0 {
							baseW = trainR.Weights
						}
						at := runCfg{
							mode: job.Mode, initW: baseW, dur: *afterTrain, win: *win, lr: job.LR,
							seed: *seed + int64(hashID(job.ID)) + 2, thinkK: *thinkK, challenge: job.Challenge,
							layer: job.Layer, dtype: job.DType, cams: job.Cams, gridN: job.GridN, train: true,
							phase: "after_train", jobID: job.ID + "|after", hub: nil,
						}.run()
						at.ID = job.ID + "|after_train"
						at.Layer, at.DType, at.LR, at.Challenge = job.Layer, job.DType.String(), job.LR, job.Challenge
						at.Cams, at.GridN = job.Cams, job.GridN
						at.Phase = "after_train"
						at.AccDelta = at.Acc - trainR.Acc
						at.ScoreDelta = at.Score - trainR.Score
						if trainR.Acc != 0 {
							trainR.ImprovePct = 100 * at.AccDelta / trainR.Acc
							at.ImprovePct = trainR.ImprovePct
						}
						trainR.Weights = at.Weights
						trainR.AccDelta = at.AccDelta
						trainR.ScoreDelta = at.ScoreDelta
						extra = append(extra, at)
					}
					outCh <- leafOut{job: job, train: trainR, extra: extra}
				}
			}()
		}

		go func() {
			for _, job := range jobs {
				if done[job.ID] {
					continue
				}
				jobCh <- job
			}
			close(jobCh)
			wg.Wait()
			close(outCh)
		}()

		treeJobs := map[string]Tree{}
		treeDone := map[string]map[string]modeResult{}
		for _, tr := range trees {
			treeJobs[tr.Key] = tr
			treeDone[tr.Key] = map[string]modeResult{}
			for _, j := range tr.Jobs {
				if done[j.ID] {
					for _, prev := range results {
						if prev.ID == j.ID && (prev.Phase == "train" || prev.Phase == "") {
							treeDone[tr.Key][j.ID] = prev
							break
						}
					}
				}
			}
		}

		finished := 0
		for out := range outCh {
			finished++
			trainR := out.train
			results = append(results, out.extra...)
			results = append(results, trainR)
			prog.DoneIDs = append(prog.DoneIDs, out.job.ID)
			done[out.job.ID] = true
			prog.Completed = append(prog.Completed, stripWeights(trainR))
			if trainR.Err == "" && (prog.BestID == "" || trainR.Score > prog.BestScore ||
				(trainR.Score == prog.BestScore && trainR.Acc > prog.BestAcc)) {
				prog.BestID = out.job.ID
				prog.BestScore = trainR.Score
				prog.BestAcc = trainR.Acc
			}
			_ = store.AppendHistory(HistoryPoint{
				At: time.Now().UTC(), JobID: out.job.ID, Phase: "train",
				Acc: trainR.Acc, Score: trainR.Score, Avail: trainR.Avail, Levels: trainR.Levels, LR: out.job.LR,
			})
			if tide != nil {
				tide.beginJob(out.job, "train", finished, pending)
				tide.finishJob(trainR)
			}
			key := treeKey(out.job)
			treeDone[key][out.job.ID] = trainR
			tr := treeJobs[key]
			if len(treeDone[key]) == len(tr.Jobs) {
				leaves := make([]modeResult, 0, len(tr.Jobs))
				for _, j := range tr.Jobs {
					leaves = append(leaves, treeDone[key][j.ID])
				}
				rep := summarizeTree(tr, leaves)
				hub.finishTree(rep)
			}
			_ = store.SaveProgress(prog)
			_ = store.SaveResults(map[string]any{
				"results": results, "reports": hub.snapshot().Reports,
				"best_id": prog.BestID, "jobs": len(jobs), "trees": len(trees), "ckpt": ckpt,
				"layer": layer, "workers": workers,
			})
			if trainR.Err != "" {
				fmt.Printf("  [%s %d/%d] ERR %s · %s\n", layer, finished, pending, shortID(out.job.ID), trainR.Err)
			} else {
				fmt.Printf("  [%s %d/%d] Acc %.1f Score %.0f · %s\n", layer, finished, pending, trainR.Acc, trainR.Score, shortID(out.job.ID))
			}
		}

		byChal := rebuildLPDByChallenge(results)
		hub.setLPDs(byChal)
		printBoard(results, byChal)
		_ = store.SaveResults(map[string]any{
			"results": results, "reports": hub.snapshot().Reports, "lpd_by_challenge": byChal,
			"best_id": prog.BestID, "jobs": len(jobs), "trees": len(trees), "ckpt": ckpt,
			"layer": layer, "workers": workers,
		})
		fmt.Printf("💾 %s/{progress,history,results}.json\n\n", ckpt)
		if *promote > 0 {
			fmt.Println("promote skipped in speed-run multi-layer pass")
		}
	}

	hub.setStatus(fmt.Sprintf("speed-run complete · layers %v", layers))
	fmt.Println("done — dash still serving; Ctrl-C to exit")
	if dash != nil {
		select {}
	}
}


func must(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
}

func hashID(id string) int {
	h := 0
	for _, c := range id {
		h = 31*h + int(c)
	}
	if h < 0 {
		h = -h
	}
	return h
}


func pickLayerInteractive() string {
	all := []string{"dense", "dense-wide", "dense-deep", "dense-deep-wide"}
	fmt.Println()
	fmt.Println("Pick layer(s) by number:")
	fmt.Println("  1) dense")
	fmt.Println("  2) dense-wide")
	fmt.Println("  3) dense-deep")
	fmt.Println("  4) dense-deep-wide")
	fmt.Println("  5) all")
	fmt.Println("  (examples: 1   |  1,3   |  1 2 4   |  all)")
	fmt.Print("> ")
	sc := bufio.NewScanner(os.Stdin)
	line := ""
	if sc.Scan() {
		line = strings.TrimSpace(sc.Text())
	}
	if line == "" {
		fmt.Println("→ 1) dense")
		return "dense"
	}
	low := strings.ToLower(line)
	if low == "all" || low == "5" {
		fmt.Println("→ 5) all (dense, dense-wide, dense-deep, dense-deep-wide)")
		return "all"
	}
	// allow spaces or commas as separators
	line = strings.ReplaceAll(line, ",", " ")
	parts := strings.Fields(line)
	var out []string
	seen := map[string]bool{}
	for _, tok := range parts {
		tok = strings.TrimSpace(tok)
		if tok == "" {
			continue
		}
		if n, err := strconv.Atoi(tok); err == nil {
			if n == 5 {
				fmt.Println("→ 5) all")
				return "all"
			}
			if n >= 1 && n <= len(all) {
				name := all[n-1]
				if !seen[name] {
					seen[name] = true
					out = append(out, name)
					fmt.Printf("  + %d) %s\n", n, name)
				}
				continue
			}
			fmt.Fprintf(os.Stderr, "unknown number %d (use 1–5)\n", n)
			continue
		}
		name := strings.ToLower(tok)
		if !seen[name] {
			seen[name] = true
			out = append(out, name)
			fmt.Printf("  + %s\n", name)
		}
	}
	if len(out) == 0 {
		fmt.Println("→ 1) dense (fallback)")
		return "dense"
	}
	return strings.Join(out, ",")
}

func resolveLayers(spec string) ([]string, error) {
	spec = strings.TrimSpace(spec)
	if spec == "" || strings.EqualFold(spec, "ask") {
		spec = pickLayerInteractive()
	}
	layers, err := parseLayerList(spec)
	if err != nil {
		return nil, err
	}
	fmt.Println("Selected layers:")
	for i, L := range layers {
		fmt.Printf("  %d/%d  %s  →  speed_ckpt/%s/\n", i+1, len(layers), L, L)
	}
	fmt.Println()
	return layers, nil
}

func (c runCfg) run() modeResult {
	return runModeInner(c)
}

func stripWeights(r modeResult) modeResult {
	r.Weights = nil
	r.Lucy.Windows = nil
	return r
}

func normalizeDashAddr(spec string) string {
	spec = strings.TrimSpace(spec)
	if spec == "" {
		return ""
	}
	// ":5151" or bare port → bind all interfaces (remote LAN access).
	if strings.HasPrefix(spec, ":") {
		return "0.0.0.0" + spec
	}
	host, port, err := net.SplitHostPort(spec)
	if err != nil {
		// bare "5151"
		if _, e := strconv.Atoi(spec); e == nil {
			return "0.0.0.0:" + spec
		}
		return spec
	}
	if host == "" || host == "localhost" {
		return "0.0.0.0:" + port
	}
	return spec
}

func dashPort(listen string) string {
	if listen == "" {
		return "5151"
	}
	if i := strings.LastIndex(listen, ":"); i >= 0 && i < len(listen)-1 {
		return listen[i+1:]
	}
	return listen
}

func shortID(id string) string {
	if len(id) <= 64 {
		return id
	}
	return id[:61] + "…"
}

func nz(s, d string) string {
	if strings.TrimSpace(s) == "" {
		return d
	}
	return s
}

func buildSweepPlan(
	modes []parallel.TrainMode, layers []string,
	dtypesSpec, challengesSpec, lrsSpec, camsSpec, gridsSpec string,
	trees, leaves, pending int, full bool,
) sweepPlan {
	modeNames := make([]string, len(modes))
	for i, m := range modes {
		modeNames[i] = m.String()
	}
	layerPart := strings.Join(layers, "+")
	modePart := strings.Join(modeNames, "+")
	if len(modeNames) > 3 {
		modePart = modeNames[0] + "+…" + fmt.Sprintf("(%d)", len(modeNames))
	}
	kind := "full permute"
	if !full {
		kind = "custom matrix"
	}
	label := fmt.Sprintf("%s × %s · %s", layerPart, modePart, kind)
	return sweepPlan{
		Label:      label,
		Modes:      modeNames,
		Layers:     append([]string(nil), layers...),
		DTypes:     dtypesSpec,
		Challenges: challengesSpec,
		LRs:        lrsSpec,
		Cams:       camsSpec,
		Grids:      gridsSpec,
		Trees:      trees,
		Leaves:     leaves,
		Pending:    pending,
		Full:       full,
	}
}

func gameLabel(game string) string {
	g := strings.TrimSpace(game)
	if g == "" || g == "mock" {
		return "mock challenges (Go)"
	}
	return g + " (bridge)"
}

func parseModeList(spec string) ([]parallel.TrainMode, error) {
	spec = strings.TrimSpace(spec)
	if spec == "" || strings.EqualFold(spec, "all") {
		return parallel.AllNamedTrainModes(), nil
	}
	var out []parallel.TrainMode
	seen := map[parallel.TrainMode]bool{}
	for _, tok := range strings.Split(spec, ",") {
		tok = strings.TrimSpace(tok)
		if tok == "" {
			continue
		}
		key := strings.ToLower(strings.ReplaceAll(strings.ReplaceAll(tok, "_", ""), "-", ""))
		var m parallel.TrainMode
		var err error
		switch key {
		case "sgd", "bp", "normal", "normalbp":
			m = parallel.ModeNormalBP
		case "stepsgd", "stepbp":
			m = parallel.ModeStepBP
		case "tween":
			m = parallel.ModeTween
		case "tweenchain":
			m = parallel.ModeTweenChain
		case "steptween":
			m = parallel.ModeStepTween
		case "steptweenchain":
			m = parallel.ModeStepTweenChain
		default:
			m, err = parallel.ParseTrainMode(tok)
			if err != nil {
				return nil, err
			}
		}
		if m == parallel.ModeInherit || seen[m] {
			continue
		}
		seen[m] = true
		out = append(out, m)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no train modes in %q", spec)
	}
	return out, nil
}

func runMode(
	mode parallel.TrainMode,
	dur, win time.Duration,
	lr float64,
	seed int64,
	thinkK int,
	game, bridgePy string,
	hub *liveHub,
) modeResult {
	return runCfg{
		mode: mode, dur: dur, win: win, lr: lr, seed: seed, thinkK: thinkK,
		game: game, challenge: chalChase, bridgePy: bridgePy,
		layer: "dense", dtype: core.DTypeFloat32, cams: 1, gridN: 1, train: true, phase: "train", hub: hub,
	}.run()
}

func runModeInner(c runCfg) modeResult {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	label := c.mode.String()
	if c.jobID != "" {
		label = c.jobID
	}
	r := modeResult{
		Mode: c.mode.String(), ThinkK: c.thinkK, Promoted: c.promoted,
		Layer: c.layer, DType: c.dtype.String(), LR: c.lr, Challenge: c.challenge,
		Cams: c.cams, GridN: c.gridN,
		Phase: c.phase, ID: c.jobID,
	}
	rng := rand.New(rand.NewSource(c.seed))
	st, err := buildPolicyNetEx(rng, core.BackendCPUTiled, c.layer, c.dtype, c.cams)
	if err != nil {
		r.Err = err.Error()
		return r
	}
	if c.initW != nil {
		if err := restoreWeights(st, c.initW); err != nil {
			r.Err = "restore: " + err.Error()
			return r
		}
	}
	r.Lucy.WeightBytes = stackWeightBytes(st)
	r.RAMKiB = float64(r.Lucy.WeightBytes) / 1024

	var grid *architecture.Grid
	if c.mode.RequiresGrid() {
		g, gerr := placeForMesh(st, c.gridN)
		if gerr != nil {
			r.Err = "place: " + gerr.Error()
			return r
		}
		grid = g
	}

	env, err := openChallengeOrGame(c.game, c.challenge, c.seed, c.bridgePy)
	if err != nil {
		r.Err = "env: " + err.Error()
		return r
	}
	defer env.Close()

	fr, err := env.Reset()
	if err != nil {
		r.Err = "reset: " + err.Error()
		return r
	}
	if c.hub != nil {
		c.hub.setFrame(fr, Action{}, label, 0)
	}

	nWin := int(c.dur / c.win)
	if nWin < 1 {
		nWin = 1
	}
	wins := make([]lucy.Window, nWin)
	var infSum, trSum time.Duration
	start := time.Now()
	maxLevels := 0

	for time.Since(start) < c.dur {
		elapsed := time.Since(start)
		wi := int(elapsed / c.win)
		if wi >= nWin {
			wi = nWin - 1
		}
		wins[wi].Phase = c.mode.Short()

		if fr.State == stateWin || fr.State == stateGameOver {
			fr, err = env.Reset()
			if err != nil {
				break
			}
		}

		tInf := startWork()
		tr, terr := thinkThenAct(st, grid, c.mode, fr, c.thinkK)
		inf := tInf.elapsed()
		infSum += inf
		if terr != nil || tr.Post == nil {
			continue
		}

		oracle := env.OracleAction(fr)
		pred := tr.Action.ID
		hard := 0.0
		if oracle >= 0 && pred == oracle {
			hard = 100
		}
		lab := oracle
		if lab < 0 {
			lab = pred
		}
		actionLogits := tr.Post.Data
		if len(actionLogits) > nActions {
			actionLogits = actionLogits[:nActions]
		}
		soft := lucy.SoftAccProb(softmaxPTrue(actionLogits, lab), 1)

		gx, gy := goalCoordsNorm(fr)
		y := targetFromOracle(lab, fr, gx, gy)

		r.Lucy.TotalOutputs++
		w := &wins[wi]
		w.Outputs++
		w.InferMs += inf.Seconds() * 1000
		w.Accuracy += hard
		w.SoftAcc += soft
		if hard == 100 {
			w.Correct++
			r.Lucy.TotalCorrect++
		}

		if c.train {
			r.Lucy.TotalTrain++
			w.TrainSteps++
			tTr := startWork()
			trainSample(st, grid, tr.MeshFwd, tr.Tape, tr.X, y, c.mode, c.lr)
			trMs := tTr.elapsed()
			trSum += trMs
			w.TrainMs += trMs.Seconds() * 1000
		}

		next, serr := env.Step(tr.Action)
		if serr != nil {
			r.Err = "step: " + serr.Error()
			break
		}
		if next.LevelsCompleted > maxLevels {
			maxLevels = next.LevelsCompleted
		}
		fr = next
		if c.hub != nil {
			c.hub.setFrame(fr, tr.Action, label, tr.ThinkK)
			secs := math.Max(time.Since(start).Seconds(), 1e-6)
			c.hub.pulseMode(modeResult{
				ID: label, Mode: c.mode.String(), Layer: c.layer, DType: c.dtype.String(),
				LR: c.lr, Challenge: c.challenge, Phase: c.phase,
				Acc: runningAcc(wins), Soft: soft,
				Avail: availOf(infSum, trSum), Thru: float64(r.Lucy.TotalOutputs) / secs,
				RAMKiB: r.RAMKiB, Levels: maxLevels, Actions: r.Lucy.TotalOutputs, ThinkK: c.thinkK,
			})
		}
	}

	for i := range wins {
		if wins[i].Outputs > 0 {
			wins[i].Accuracy /= float64(wins[i].Outputs)
			wins[i].SoftAcc /= float64(wins[i].Outputs)
		}
	}
	r.Lucy.Windows = wins
	r.Lucy.Duration = time.Since(start)
	r.Lucy.InferMs = infSum.Seconds() * 1000
	r.Lucy.TrainMs = trSum.Seconds() * 1000
	lucy.Finalize(&r.Lucy, lucy.Options{AdaptWindows: 1, ConsThreshold: lucy.ConsThreshold})
	r.Acc = r.Lucy.AvgAccuracy
	r.Soft = r.Lucy.SoftAcc
	r.Avail = r.Lucy.Availability
	r.Thru = r.Lucy.Throughput
	r.Score = r.Lucy.Score
	r.Levels = maxLevels
	r.Actions = r.Lucy.TotalOutputs
	r.Weights = weightSnapshot(st)
	if c.hub != nil {
		c.hub.finishMode(r)
	}
	return r
}

func runningAcc(wins []lucy.Window) float64 {
	var o, c float64
	for _, w := range wins {
		o += float64(w.Outputs)
		c += float64(w.Correct)
	}
	if o == 0 {
		return 0
	}
	return 100 * c / o
}

func availOf(inf, tr time.Duration) float64 {
	t := inf + tr
	if t <= 0 {
		return 100 // freeze phase: all infer
	}
	return 100 * float64(inf) / float64(t)
}


func lpdSample(r modeResult) (lucy.Sample, bool) {
	phaseOK := r.Phase == "" || r.Phase == "train" || r.Phase == "after_train"
	if r.Err != "" || !phaseOK {
		return lucy.Sample{}, false
	}
	id := r.Mode
	if r.ID != "" {
		id = r.ID
	}
	arch := nz(r.Layer, "dense-think")
	if r.Cams > 1 {
		arch = fmt.Sprintf("%s|%s|%s", arch, camName(r.Cams), gridName(r.GridN))
	} else if r.GridN > 1 {
		arch = fmt.Sprintf("%s|%s", arch, gridName(r.GridN))
	}
	return lucy.Sample{
		ID: id, Mode: r.Mode, DType: nz(r.DType, "float32"), Arch: arch,
		Score: r.Score, Soft: r.Soft, Acc: r.Acc, Thru: r.Thru, Avail: r.Avail, RAMKiB: r.RAMKiB,
	}, true
}

// rebuildLPDByChallenge builds a separate LPD board per challenge so chase Acc
// never ranks against teleport Acc (apples-to-oranges).
func rebuildLPDByChallenge(rows []modeResult) map[string]lucy.LPD {
	buckets := map[string][]lucy.Sample{}
	for _, r := range rows {
		s, ok := lpdSample(r)
		if !ok {
			continue
		}
		chal := r.Challenge
		if chal == "" {
			chal = "unknown"
		}
		buckets[chal] = append(buckets[chal], s)
	}
	out := make(map[string]lucy.LPD, len(buckets))
	for chal, pts := range buckets {
		out[chal] = lucy.BuildLPD(pts)
	}
	return out
}

func orderedLPDChallenges(by map[string]lucy.LPD) []string {
	var out []string
	seen := map[string]bool{}
	for _, c := range allChallenges() {
		if _, ok := by[c]; ok {
			out = append(out, c)
			seen[c] = true
		}
	}
	var extra []string
	for c := range by {
		if !seen[c] {
			extra = append(extra, c)
		}
	}
	sort.Strings(extra)
	return append(out, extra...)
}

func printBoard(rows []modeResult, byChal map[string]lucy.LPD) {
	for _, chal := range orderedLPDChallenges(byChal) {
		fmt.Printf("\n── LPD · %s ──\n", chal)
		fmt.Println("╔══════════════════════╦═══════╦═══════╦═══════╦════════╦════════╗")
		fmt.Println("║ Job / Mode           ║ Acc   ║ Soft  ║ Avail ║ Score  ║ Lv ║ Δacc ║")
		fmt.Println("╠══════════════════════╬═══════╬═══════╬═══════╬════════╬════════╣")
		n := 0
		for _, r := range rows {
			if r.Challenge != chal {
				continue
			}
			n++
			name := r.Mode
			if r.ID != "" {
				name = shortID(r.ID)
			}
			if r.Err != "" {
				fmt.Printf("║ %-20s ║ ERR %s\n", clip(name, 20), r.Err)
				continue
			}
			fmt.Printf("║ %-20s ║ %5.1f ║ %5.1f ║ %5.1f ║ %6.0f ║ %2d ║ %+5.1f ║\n",
				clip(name, 20), r.Acc, r.Soft, r.Avail, r.Score, r.Levels, r.AccDelta)
		}
		if n == 0 {
			fmt.Println("║ (no rows)            ║")
		}
		fmt.Println("╚══════════════════════╩═══════╩═══════╩═══════╩════════╩════════╝")
		lpd := byChal[chal]
		fmt.Printf("LPD champ=%s live=%s gold-std=%s (n=%d)\n",
			lpd.Champ.Mode, lpd.LiveChamp.Mode, lpd.GoldStd.Mode, lpd.N)
	}
}

func clip(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n-1] + "…"
}
