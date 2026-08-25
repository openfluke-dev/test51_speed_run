package main

import (
	"fmt"
	"strings"
)

// Challenge IDs for the mock ARC-shaped suite.
const (
	chalChase    = "chase"    // reach goal
	chalFlee     = "flee"     // stay away from hunter
	chalCollect  = "collect"  // hit N waypoints (chase levels)
	chalTeleport = "teleport" // goal jumps when approached
	chalShock    = "shock"    // mid-episode oracle flips (chase↔flee)
)

func allChallenges() []string {
	return []string{chalChase, chalFlee, chalCollect, chalTeleport, chalShock}
}

func parseChallengeList(spec string) ([]string, error) {
	spec = strings.TrimSpace(spec)
	if spec == "" || strings.EqualFold(spec, "all") {
		return allChallenges(), nil
	}
	if strings.EqualFold(spec, "mock") {
		return []string{chalChase}, nil
	}
	want := map[string]bool{}
	for _, c := range allChallenges() {
		want[c] = true
	}
	var out []string
	seen := map[string]bool{}
	for _, tok := range strings.Split(spec, ",") {
		tok = strings.ToLower(strings.TrimSpace(tok))
		if tok == "" {
			continue
		}
		if !want[tok] {
			return nil, fmt.Errorf("unknown challenge %q (chase|flee|collect|teleport|shock|all)", tok)
		}
		if seen[tok] {
			continue
		}
		seen[tok] = true
		out = append(out, tok)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("no challenges in %q", spec)
	}
	return out, nil
}

// ChallengeEnv wraps MockEnv with different goals / oracles.
type ChallengeEnv struct {
	inner     *MockEnv
	kind      string
	shockFlip bool
	shockAt   int
}

func newChallengeEnv(kind string, seed int64) *ChallengeEnv {
	m := NewMockEnv(seed, "mock-"+kind)
	ce := &ChallengeEnv{inner: m, kind: kind}
	switch kind {
	case chalFlee:
		m.winLevels = 5
		m.maxSteps = 60
	case chalCollect:
		m.winLevels = 4
	case chalShock:
		ce.shockAt = m.maxSteps / 2
	}
	return ce
}

func (c *ChallengeEnv) Close() error { return c.inner.Close() }

func (c *ChallengeEnv) Reset() (Frame, error) {
	c.shockFlip = false
	fr, err := c.inner.Reset()
	if err != nil {
		return fr, err
	}
	fr.GameID = "mock-" + c.kind
	return fr, nil
}

func (c *ChallengeEnv) Step(a Action) (Frame, error) {
	m := c.inner
	if c.kind == chalShock && m.steps+1 >= c.shockAt {
		c.shockFlip = true
	}

	switch c.kind {
	case chalFlee:
		return c.stepFlee(a)
	case chalTeleport:
		return c.stepTeleport(a)
	case chalShock:
		if c.shockFlip {
			return c.stepFlee(a)
		}
		fr, err := m.Step(a)
		if err == nil {
			fr.GameID = "mock-shock"
		}
		return fr, err
	default: // chase, collect
		fr, err := m.Step(a)
		if err == nil {
			fr.GameID = "mock-" + c.kind
		}
		return fr, err
	}
}

func (c *ChallengeEnv) OracleAction(fr Frame) int {
	m := c.inner
	switch c.kind {
	case chalFlee:
		return c.oracleFlee()
	case chalShock:
		if c.shockFlip {
			return c.oracleFlee()
		}
		return m.OracleAction(fr)
	default:
		return m.OracleAction(fr)
	}
}

func (c *ChallengeEnv) oracleFlee() int {
	m := c.inner
	dx := m.gx - m.ax
	dy := m.gy - m.ay
	if abs(dx)+abs(dy) == 0 {
		return 5
	}
	// move opposite to hunter (goal cell acts as hunter)
	if abs(dx) >= abs(dy) {
		if dx < 0 {
			return 4
		}
		return 3
	}
	if dy < 0 {
		return 2
	}
	return 1
}

func (c *ChallengeEnv) applyAgentMove(a Action) {
	m := c.inner
	if a.ID == 0 {
		m.spawn()
		m.steps = 0
		return
	}
	m.steps++
	switch a.ID {
	case 1:
		m.ay = clamp(m.ay-1, 0, frameSize-1)
	case 2:
		m.ay = clamp(m.ay+1, 0, frameSize-1)
	case 3:
		m.ax = clamp(m.ax-1, 0, frameSize-1)
	case 4:
		m.ax = clamp(m.ax+1, 0, frameSize-1)
	case 5, 7:
	case 6:
		m.ax = clamp(a.X, 0, frameSize-1)
		m.ay = clamp(a.Y, 0, frameSize-1)
	}
}

func (c *ChallengeEnv) stepHunter() {
	m := c.inner
	if m.ax < m.gx {
		m.gx--
	} else if m.ax > m.gx {
		m.gx++
	} else if m.ay < m.gy {
		m.gy--
	} else if m.ay > m.gy {
		m.gy++
	}
}

func (c *ChallengeEnv) stepFlee(a Action) (Frame, error) {
	m := c.inner
	c.applyAgentMove(a)
	c.stepHunter()
	gid := "mock-" + c.kind
	if m.ax == m.gx && m.ay == m.gy {
		m.spawn()
		m.steps = 0
		fr := m.frame(stateGameOver)
		fr.GameID = gid
		return fr, nil
	}
	if m.steps >= m.maxSteps {
		m.levels++
		if m.levels >= m.winLevels {
			fr := m.frame(stateWin)
			fr.GameID = gid
			return fr, nil
		}
		m.spawn()
		m.steps = 0
		fr := m.frame(stateNotFinished)
		fr.GameID = gid
		return fr, nil
	}
	fr := m.frame(stateNotFinished)
	fr.GameID = gid
	return fr, nil
}

func (c *ChallengeEnv) stepTeleport(a Action) (Frame, error) {
	m := c.inner
	fr, err := m.Step(a)
	if err != nil {
		return fr, err
	}
	if abs(m.gx-m.ax)+abs(m.gy-m.ay) <= 3 && fr.State == stateNotFinished {
		m.gx = 8 + m.rng.Intn(frameSize-16)
		m.gy = 8 + m.rng.Intn(frameSize-16)
		fr = m.frame(stateNotFinished)
	}
	fr.GameID = "mock-teleport"
	return fr, nil
}

func openChallengeOrGame(game, challenge string, seed int64, bridgePy string) (Env, error) {
	g := strings.TrimSpace(game)
	if g != "" && g != "mock" {
		return openEnv(g, seed, bridgePy)
	}
	ch := strings.TrimSpace(challenge)
	if ch == "" {
		ch = chalChase
	}
	return newChallengeEnv(ch, seed), nil
}
