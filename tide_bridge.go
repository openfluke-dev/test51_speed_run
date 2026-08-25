package main

import (
	"fmt"
	"os"
	"strings"

	"github.com/openfluke/tide/dash"
	"github.com/openfluke/tide/permute"
	"github.com/openfluke/tide/pulse"
	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/parallel"
	"github.com/openfluke/welvet/lucy"
	"github.com/openfluke/welvet/quant"
)

// tideBridge runs Tide's Lucy dash beside test51's tree dash.
// test51 → :5151 (tree consolidations); Tide → :8080 (Lucy /api/report.pdf).
type tideBridge struct {
	tr  *pulse.Tracker
	srv *dash.Server
}

func startTideBridge(addr string, jobs []Job, lr float64, task string) *tideBridge {
	addr = strings.TrimSpace(addr)
	if addr == "" {
		return nil
	}
	listen := normalizeDashAddr(addr)
	cells := make([]permute.Cell, 0, len(jobs))
	seen := map[string]bool{}
	for _, j := range jobs {
		c := jobToTideCell(j)
		if seen[c.ID] {
			continue
		}
		seen[c.ID] = true
		cells = append(cells, c)
	}
	tr := pulse.New()
	srv := &dash.Server{
		Tracker:  tr,
		Cells:    cells,
		Addr:     listen,
		Epoch:    1,
		Task:     nz(task, "test51"),
		Subtitle: "think-race host · tree sweep feeds Tide Lucy reports",
		LR:       lr,
		ID:       "test51",
	}
	go func() {
		if err := srv.ListenAndServe(); err != nil {
			fmt.Fprintf(os.Stderr, "tide dash: %v\n", err)
		}
	}()
	port := dashPort(listen)
	fmt.Printf("tide listening on %s  →  http://<this-host-ip>:%s  (Lucy board + /api/report.pdf)\n", listen, port)
	return &tideBridge{tr: tr, srv: srv}
}

func (t *tideBridge) signalStart() {
	if t == nil || t.srv == nil {
		return
	}
	t.srv.SignalStart()
}

func (t *tideBridge) setProgress(idx, total int, msg string) {
	if t == nil || t.tr == nil {
		return
	}
	t.tr.SetCellProgress(idx, total, msg)
}

func (t *tideBridge) beginJob(job Job, phase string, idx, total int) {
	if t == nil || t.tr == nil {
		return
	}
	t.tr.SetCellProgress(idx, total, phase+" · "+shortID(job.ID))
	t.tr.Begin(jobToTideCell(job), phase)
}

func (t *tideBridge) pulseRunning(r modeResult) {
	if t == nil || t.tr == nil {
		return
	}
	snap := lucy.Snapshot{
		AvgAccuracy:  r.Acc,
		SoftAcc:      r.Soft,
		Availability: r.Avail,
		Throughput:   r.Thru,
		Score:        r.Score,
		TotalOutputs: r.Actions,
	}
	if r.RAMKiB > 0 {
		snap.WeightBytes = int64(r.RAMKiB * 1024)
	}
	t.tr.Pulse(lucy.Window{}, snap, nz(r.Phase, "train"))
}

func (t *tideBridge) finishJob(r modeResult) {
	if t == nil || t.tr == nil {
		return
	}
	status := "ok"
	note := r.Challenge
	if r.Err != "" {
		status = "fail"
		note = r.Err
	}
	snap := r.Lucy
	if snap.Score == 0 && r.Score != 0 {
		snap.AvgAccuracy = r.Acc
		snap.SoftAcc = r.Soft
		snap.Availability = r.Avail
		snap.Throughput = r.Thru
		snap.Score = r.Score
		snap.TotalOutputs = r.Actions
		if r.RAMKiB > 0 {
			snap.WeightBytes = int64(r.RAMKiB * 1024)
		}
	}
	t.tr.Finish(status, note, snap)
}

// seedCompleted replays finished train leaves into Tide so the board/PDF
// show resume history before Start.
func (t *tideBridge) seedCompleted(rows []modeResult) {
	if t == nil || t.tr == nil {
		return
	}
	n := 0
	for _, r := range rows {
		phaseOK := r.Phase == "" || r.Phase == "train" || r.Phase == "after_train"
		if r.Err != "" || !phaseOK {
			continue
		}
		cell := resultToTideCell(r)
		t.tr.Begin(cell, "train")
		t.tr.Finish("ok", r.Challenge, r.Lucy)
		n++
	}
	if n > 0 {
		fmt.Printf("tide: seeded %d completed cell(s) from ckpt\n", n)
	}
}

func jobToTideCell(j Job) permute.Cell {
	cams := j.Cams
	if cams < 1 {
		cams = 1
	}
	return permute.Cell{
		ID:      j.ID,
		DType:   j.DType,
		Format:  quant.FormatNone,
		Mode:    welvetModeToTide(j.Mode),
		Arch:    permute.ArchForCams(cams),
		Cams:    cams,
		Backend: core.BackendSIMD,
		UseSIMD: true,
	}
}

func resultToTideCell(r modeResult) permute.Cell {
	cams := r.Cams
	if cams < 1 {
		cams = 1
	}
	dt := core.ParseDType(r.DType)
	mode, err := parallel.ParseTrainMode(r.Mode)
	if err != nil {
		mode = parallel.ModeNormalBP
	}
	id := r.ID
	if id == "" {
		id = r.Mode
	}
	// Strip phase suffixes so resume seed matches job IDs.
	for _, suf := range []string{"|after_train", "|after_freeze", "|freeze", "|after"} {
		if len(id) > len(suf) && id[len(id)-len(suf):] == suf {
			id = id[:len(id)-len(suf)]
			break
		}
	}
	return permute.Cell{
		ID:      id,
		DType:   dt,
		Format:  quant.FormatNone,
		Mode:    welvetModeToTide(mode),
		Arch:    permute.ArchForCams(cams),
		Cams:    cams,
		Backend: core.BackendSIMD,
		UseSIMD: true,
	}
}

func welvetModeToTide(m parallel.TrainMode) permute.TrainMode {
	switch m {
	case parallel.ModeNormalBP:
		return permute.ModeSGD
	case parallel.ModeStepBP:
		return permute.ModeStepSGD
	case parallel.ModeTween:
		return permute.ModeTween
	case parallel.ModeTweenChain:
		return permute.ModeTweenChain
	case parallel.ModeStepTween:
		return permute.ModeStepTween
	case parallel.ModeStepTweenChain:
		return permute.ModeStepTweenChain
	default:
		return permute.TrainMode(m.String())
	}
}
