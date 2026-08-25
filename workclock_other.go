//go:build !linux

package main

import "time"

func threadCPU() (time.Duration, bool) {
	return 0, false
}

func dutyClockName() string {
	return "wall-clock"
}
