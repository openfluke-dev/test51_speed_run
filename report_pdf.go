package main

import (
	"bytes"
	"fmt"
	"math"
	"sort"
	"strings"
)

// buildReportPDF creates a simple PDF with summary, Acc-by-LR bars, heatmap, and leaf table.
func buildReportPDF(rep treeReport) []byte {
	var content bytes.Buffer
	pageW, pageH := 612.0, 792.0 // US Letter
	y := pageH - 48

	write := func(s string) { content.WriteString(s) }
	text := func(x, yy float64, size float64, s string) {
		s = pdfEscape(s)
		write(fmt.Sprintf("BT /F1 %.1f Tf %.1f %.1f Td (%s) Tj ET\n", size, x, yy, s))
	}

	// Header
	text(48, y, 18, fmt.Sprintf("test51 tree report #%d", rep.Index))
	y -= 22
	text(48, y, 11, fmt.Sprintf("%s  ·  %s/%s  ·  %s  ·  %d leaves",
		rep.Mode, rep.Layer, rep.DType, rep.Challenge, rep.Leaves))
	y -= 16
	text(48, y, 10, fmt.Sprintf("Finished %s", rep.Finished.UTC().Format("2006-01-02 15:04:05 UTC")))
	y -= 28

	text(48, y, 12, fmt.Sprintf("Best Acc %.1f   Score %.0f   Δacc %+.1f   lr=%g cam=%d grid=%d³",
		rep.BestAcc, rep.BestScore, rep.BestΔ, rep.BestLR, rep.BestCams, rep.BestGrid))
	y -= 18
	text(48, y, 10, "Best leaf: "+rep.BestID)
	y -= 28

	rows := append([]leafRow(nil), rep.Rows...)
	sort.Slice(rows, func(i, j int) bool { return rows[i].Score > rows[j].Score })

	// Acc by LR bars
	text(48, y, 12, "Accuracy by learning rate (avg)")
	y -= 8
	lrPts := avgMetric(rows, func(r leafRow) float64 { return r.LR }, func(r leafRow) float64 { return r.Acc })
	barH := 90.0
	drawHBars(&content, 48, y-barH, 516, barH, lrPts, 0.24, 0.84, 0.78)
	y -= barH + 24

	// Score heatmap cams × grid
	text(48, y, 12, "Score heatmap · cams × grid (avg)")
	y -= 8
	heatH := 110.0
	drawHeatPDF(&content, 48, y-heatH, 516, heatH, rows)
	y -= heatH + 28

	// Leaf table (top rows that fit)
	text(48, y, 12, "Leaves by score")
	y -= 16
	text(48, y, 9, "rank  lr        cam  grid   acc    soft   score   Δacc    avail")
	y -= 12
	write("0.4 0.45 0.5 RG 48 " + fmt.Sprintf("%.1f", y+4) + " m 564 " + fmt.Sprintf("%.1f", y+4) + " l S\n")

	maxRows := 28
	for i, r := range rows {
		if i >= maxRows || y < 56 {
			text(48, y-12, 9, fmt.Sprintf("… %d more leaves (open HTML report for full table)", len(rows)-i))
			break
		}
		line := fmt.Sprintf("%3d  %-8s  %3d  %3d³  %5.1f  %5.1f  %6.0f  %+5.1f  %5.1f",
			i+1, fmtLR(r.LR), r.Cams, maxInt(r.GridN, 1), r.Acc, r.Soft, r.Score, r.AccΔ, r.Avail)
		text(48, y, 8, line)
		y -= 11
	}

	stream := content.Bytes()
	return assemblePDF(stream, int(pageW), int(pageH))
}

func fmtLR(lr float64) string {
	if lr >= 100 {
		return fmt.Sprintf("%.0f", lr)
	}
	if lr >= 1 {
		return fmt.Sprintf("%.2f", lr)
	}
	return fmt.Sprintf("%g", lr)
}

func avgMetric(rows []leafRow, key func(leafRow) float64, val func(leafRow) float64) [][2]float64 {
	type agg struct{ sum, n float64 }
	m := map[float64]*agg{}
	var keys []float64
	for _, r := range rows {
		if r.Err != "" {
			continue
		}
		k := key(r)
		a := m[k]
		if a == nil {
			a = &agg{}
			m[k] = a
			keys = append(keys, k)
		}
		a.sum += val(r)
		a.n++
	}
	sort.Float64s(keys)
	out := make([][2]float64, 0, len(keys))
	for _, k := range keys {
		a := m[k]
		out = append(out, [2]float64{k, a.sum / a.n})
	}
	return out
}

func drawHBars(buf *bytes.Buffer, x, y, w, h float64, pts [][2]float64, r, g, b float64) {
	write := func(s string) { buf.WriteString(s) }
	write("0.15 0.18 0.22 rg\n")
	write(fmt.Sprintf("%.1f %.1f %.1f %.1f re f\n", x, y, w, h))
	if len(pts) == 0 {
		return
	}
	maxV := 1.0
	for _, p := range pts {
		if p[1] > maxV {
			maxV = p[1]
		}
	}
	bw := (w - 16) / float64(len(pts))
	for i, p := range pts {
		bh := (h - 20) * (p[1] / maxV)
		bx := x + 8 + float64(i)*bw + 2
		by := y + 12
		write(fmt.Sprintf("%.3f %.3f %.3f rg\n", r, g, b))
		write(fmt.Sprintf("%.1f %.1f %.1f %.1f re f\n", bx, by, math.Max(bw-4, 2), math.Max(bh, 1)))
	}
}

func drawHeatPDF(buf *bytes.Buffer, x, y, w, h float64, rows []leafRow) {
	write := func(s string) { buf.WriteString(s) }
	write("0.15 0.18 0.22 rg\n")
	write(fmt.Sprintf("%.1f %.1f %.1f %.1f re f\n", x, y, w, h))

	camSet, gridSet := map[int]bool{}, map[int]bool{}
	type agg struct{ sum, n float64 }
	cell := map[[2]int]*agg{}
	maxV := 1.0
	for _, r := range rows {
		if r.Err != "" {
			continue
		}
		c, g := r.Cams, r.GridN
		if c < 1 {
			c = 1
		}
		if g < 1 {
			g = 1
		}
		camSet[c], gridSet[g] = true, true
		k := [2]int{c, g}
		a := cell[k]
		if a == nil {
			a = &agg{}
			cell[k] = a
		}
		a.sum += r.Score
		a.n++
		if v := a.sum / a.n; v > maxV {
			maxV = v
		}
	}
	cams, grids := keysInt(camSet), keysInt(gridSet)
	if len(cams) == 0 || len(grids) == 0 {
		return
	}
	cw := (w - 40) / float64(len(grids))
	ch := (h - 24) / float64(len(cams))
	for yi, c := range cams {
		for xi, g := range grids {
			a := cell[[2]int{c, g}]
			v := 0.0
			if a != nil && a.n > 0 {
				v = a.sum / a.n
			}
			t := v / maxV
			write(fmt.Sprintf("%.3f %.3f %.3f rg\n", 0.1+0.14*t, 0.3+0.54*t, 0.28+0.5*t))
			write(fmt.Sprintf("%.1f %.1f %.1f %.1f re f\n",
				x+36+float64(xi)*cw+2, y+16+float64(len(cams)-1-yi)*ch+2, cw-4, ch-4))
		}
	}
}

func keysInt(m map[int]bool) []int {
	out := make([]int, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Ints(out)
	return out
}

func pdfEscape(s string) string {
	s = strings.ReplaceAll(s, "\\", "\\\\")
	s = strings.ReplaceAll(s, "(", "\\(")
	s = strings.ReplaceAll(s, ")", "\\)")
	s = strings.Map(func(r rune) rune {
		if r < 32 || r > 126 {
			return '?'
		}
		return r
	}, s)
	return s
}

func assemblePDF(stream []byte, pageW, pageH int) []byte {
	var out bytes.Buffer
	offsets := []int{0} // 1-indexed objects; placeholder

	writeObj := func(body string) {
		offsets = append(offsets, out.Len())
		fmt.Fprintf(&out, "%d 0 obj\n%s\nendobj\n", len(offsets)-1, body)
	}

	out.WriteString("%PDF-1.4\n%\xff\xff\xff\xff\n")

	// 1: Catalog
	writeObj("<< /Type /Catalog /Pages 2 0 R >>")
	// 2: Pages
	writeObj("<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
	// 3: Page
	writeObj(fmt.Sprintf(
		"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %d %d] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>",
		pageW, pageH))
	// 4: Content stream
	offsets = append(offsets, out.Len())
	fmt.Fprintf(&out, "4 0 obj\n<< /Length %d >>\nstream\n", len(stream))
	out.Write(stream)
	out.WriteString("\nendstream\nendobj\n")
	// 5: Font
	writeObj("<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")

	xref := out.Len()
	fmt.Fprintf(&out, "xref\n0 %d\n", len(offsets))
	out.WriteString("0000000000 65535 f \n")
	for i := 1; i < len(offsets); i++ {
		fmt.Fprintf(&out, "%010d 00000 n \n", offsets[i])
	}
	fmt.Fprintf(&out, "trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n", len(offsets), xref)
	return out.Bytes()
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
