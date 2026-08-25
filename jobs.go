package main

import (
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/openfluke/welvet/core"
	"github.com/openfluke/welvet/layers/parallel"
)

// Job is one cartesian cell.
//
// Expand order (outer → inner):
//
//	mode → layer → dtype → challenge → LR↑ → cams → grid
//
// One "tree" = fixed (mode, layer, dtype, challenge). Leaves inside a tree sweep
// LR↑ × cams × grids so you finish that architecture's variants before the next mode.
type Job struct {
	ID        string
	Mode      parallel.TrainMode
	Layer     string
	DType     core.DType
	LR        float64
	Challenge string
	Cams      int // 1 = single, 2 = bicameral, 3 = tricameral
	GridN     int // mesh cube edge: 1 → 1×1×1 … 3 → 3×3×3
}

// Tree is one architecture slice: fixed train-mode + layer + dtype + challenge.
// Leaves are the LR × cam × grid variants (board shows only the active tree).
type Tree struct {
	Key       string
	Mode      string
	Layer     string
	DType     string
	Challenge string
	Jobs      []Job
}

var defaultFunnyLRs = []float64{0.02, 0.2, 2, 20, 200, 1000, 10000, 1_000_000}

func parseLRList(spec string) ([]float64, error) {
	spec = strings.TrimSpace(spec)
	if spec == "" || strings.EqualFold(spec, "funny") || strings.EqualFold(spec, "all") {
		out := make([]float64, len(defaultFunnyLRs))
		copy(out, defaultFunnyLRs)
		return out, nil
	}
	var out []float64
	for _, tok := range strings.Split(spec, ",") {
		tok = strings.TrimSpace(tok)
		if tok == "" {
			continue
		}
		v, err := strconv.ParseFloat(tok, 64)
		if err != nil {
			return nil, fmt.Errorf("lr %q: %w", tok, err)
		}
		out = append(out, v)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no learning rates in %q", spec)
	}
	sort.Float64s(out)
	return out, nil
}

func parseDTypeList(spec string) ([]core.DType, error) {
	spec = strings.TrimSpace(spec)
	if spec == "" {
		return []core.DType{core.DTypeFloat32}, nil
	}
	if strings.EqualFold(spec, "all") {
		out := make([]core.DType, len(core.AllDTypes))
		copy(out, core.AllDTypes)
		return out, nil
	}
	var out []core.DType
	seen := map[core.DType]bool{}
	for _, tok := range strings.Split(spec, ",") {
		tok = strings.TrimSpace(tok)
		if tok == "" {
			continue
		}
		var dt core.DType
		switch strings.ToLower(tok) {
		case "f32", "float":
			dt = core.DTypeFloat32
		case "f64", "double":
			dt = core.DTypeFloat64
		case "f16":
			dt = core.DTypeFloat16
		case "bf16":
			dt = core.DTypeBFloat16
		default:
			ok := false
			for _, known := range core.AllDTypes {
				if strings.EqualFold(known.String(), tok) {
					dt = known
					ok = true
					break
				}
			}
			if !ok {
				dt = core.ParseDType(tok)
				if !strings.EqualFold(dt.String(), tok) &&
					!strings.EqualFold(tok, "f32") &&
					!strings.EqualFold(tok, "float32") {
					return nil, fmt.Errorf("unknown dtype %q", tok)
				}
			}
		}
		if seen[dt] {
			continue
		}
		seen[dt] = true
		out = append(out, dt)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no dtypes in %q", spec)
	}
	return out, nil
}

func parseLayerList(spec string) ([]string, error) {
	spec = strings.TrimSpace(spec)
	if spec == "" {
		return []string{"dense"}, nil
	}
	if strings.EqualFold(spec, "all") {
		return []string{"dense", "dense-wide", "dense-deep", "dense-deep-wide"}, nil
	}
	var out []string
	seen := map[string]bool{}
	for _, tok := range strings.Split(spec, ",") {
		tok = strings.ToLower(strings.TrimSpace(tok))
		if tok == "" {
			continue
		}
		switch tok {
		case "dense", "dense-wide", "dense-deep", "dense-deep-wide":
		default:
			return nil, fmt.Errorf("unknown layer recipe %q (dense|dense-wide|dense-deep|dense-deep-wide|all)", tok)
		}
		if seen[tok] {
			continue
		}
		seen[tok] = true
		out = append(out, tok)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no layers in %q", spec)
	}
	return out, nil
}

// parseCamsList accepts "1", "1,2,3", "1-3", "all".
func parseCamsList(spec string) ([]int, error) {
	spec = strings.TrimSpace(spec)
	if spec == "" {
		return []int{1}, nil
	}
	if strings.EqualFold(spec, "all") || spec == "1-3" {
		return []int{1, 2, 3}, nil
	}
	var out []int
	seen := map[int]bool{}
	for _, tok := range strings.Split(spec, ",") {
		tok = strings.TrimSpace(tok)
		if tok == "" {
			continue
		}
		if strings.Contains(tok, "-") {
			parts := strings.SplitN(tok, "-", 2)
			lo, err1 := strconv.Atoi(strings.TrimSpace(parts[0]))
			hi, err2 := strconv.Atoi(strings.TrimSpace(parts[1]))
			if err1 != nil || err2 != nil || lo < 1 || hi > 3 || lo > hi {
				return nil, fmt.Errorf("cams range %q (want 1-3)", tok)
			}
			for n := lo; n <= hi; n++ {
				if !seen[n] {
					seen[n] = true
					out = append(out, n)
				}
			}
			continue
		}
		n, err := strconv.Atoi(tok)
		if err != nil || n < 1 || n > 3 {
			return nil, fmt.Errorf("cams %q (want 1..3)", tok)
		}
		if seen[n] {
			continue
		}
		seen[n] = true
		out = append(out, n)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no cams in %q", spec)
	}
	sort.Ints(out)
	return out, nil
}

// parseGridList accepts "1", "1,2,3", "1x1x1,3x3x3", "1-3", "all".
func parseGridList(spec string) ([]int, error) {
	spec = strings.TrimSpace(spec)
	if spec == "" {
		return []int{1}, nil
	}
	if strings.EqualFold(spec, "all") || spec == "1-3" {
		return []int{1, 2, 3}, nil
	}
	var out []int
	seen := map[int]bool{}
	for _, tok := range strings.Split(spec, ",") {
		tok = strings.ToLower(strings.TrimSpace(tok))
		if tok == "" {
			continue
		}
		if strings.Contains(tok, "-") && !strings.Contains(tok, "x") {
			parts := strings.SplitN(tok, "-", 2)
			lo, err1 := strconv.Atoi(strings.TrimSpace(parts[0]))
			hi, err2 := strconv.Atoi(strings.TrimSpace(parts[1]))
			if err1 != nil || err2 != nil || lo < 1 || hi > 3 || lo > hi {
				return nil, fmt.Errorf("grids range %q (want 1-3)", tok)
			}
			for n := lo; n <= hi; n++ {
				if !seen[n] {
					seen[n] = true
					out = append(out, n)
				}
			}
			continue
		}
		n := 0
		if strings.Contains(tok, "x") {
			parts := strings.Split(tok, "x")
			if len(parts) != 3 {
				return nil, fmt.Errorf("grid %q (want NxNxN)", tok)
			}
			a, e1 := strconv.Atoi(parts[0])
			b, e2 := strconv.Atoi(parts[1])
			c, e3 := strconv.Atoi(parts[2])
			if e1 != nil || e2 != nil || e3 != nil || a != b || b != c || a < 1 || a > 3 {
				return nil, fmt.Errorf("grid %q (want 1x1x1..3x3x3)", tok)
			}
			n = a
		} else {
			var err error
			n, err = strconv.Atoi(tok)
			if err != nil || n < 1 || n > 3 {
				return nil, fmt.Errorf("grid %q (want 1..3)", tok)
			}
		}
		if seen[n] {
			continue
		}
		seen[n] = true
		out = append(out, n)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no grids in %q", spec)
	}
	sort.Ints(out)
	return out, nil
}

func camName(n int) string {
	switch n {
	case 1:
		return "single"
	case 2:
		return "bicameral"
	case 3:
		return "tricameral"
	default:
		return fmt.Sprintf("%d-cam", n)
	}
}

func gridName(n int) string {
	if n < 1 {
		n = 1
	}
	return fmt.Sprintf("%dx%dx%d", n, n, n)
}

func expandJobs(
	modes []parallel.TrainMode,
	layers []string,
	dtypes []core.DType,
	lrs []float64,
	challenges []string,
	cams []int,
	grids []int,
) []Job {
	if len(cams) == 0 {
		cams = []int{1}
	}
	if len(grids) == 0 {
		grids = []int{1}
	}
	// Always climb LRs low → high (funny sweep reads as a learning-rate ramp).
	sorted := append([]float64(nil), lrs...)
	sort.Float64s(sorted)

	var jobs []Job
	// Outer: architecture identity. Inner: LR↑ × cams × grids (finish one tree first).
	for _, mode := range modes {
		for _, layer := range layers {
			for _, dt := range dtypes {
				for _, ch := range challenges {
					for _, lr := range sorted {
						for _, cam := range cams {
							for _, gn := range grids {
								id := fmt.Sprintf("%s|%s|%s|%s|lr=%g|cam=%d|%s",
									ch, gridName(gn), layer, dt.String(), lr, cam, mode.String())
								jobs = append(jobs, Job{
									ID:        id,
									Mode:      mode,
									Layer:     layer,
									DType:     dt,
									LR:        lr,
									Challenge: ch,
									Cams:      cam,
									GridN:     gn,
								})
							}
						}
					}
				}
			}
		}
	}
	return jobs
}

func treeKey(j Job) string {
	return fmt.Sprintf("%s|%s|%s|%s", j.Mode.String(), j.Layer, j.DType.String(), j.Challenge)
}

// groupTrees collapses a flat job list into architecture trees (preserves expand order).
func groupTrees(jobs []Job) []Tree {
	var trees []Tree
	idx := map[string]int{}
	for _, j := range jobs {
		k := treeKey(j)
		if i, ok := idx[k]; ok {
			trees[i].Jobs = append(trees[i].Jobs, j)
			continue
		}
		idx[k] = len(trees)
		trees = append(trees, Tree{
			Key:       k,
			Mode:      j.Mode.String(),
			Layer:     j.Layer,
			DType:     j.DType.String(),
			Challenge: j.Challenge,
			Jobs:      []Job{j},
		})
	}
	return trees
}
