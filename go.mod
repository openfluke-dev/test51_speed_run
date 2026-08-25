module github.com/openfluke/welvet/apps/aai/test51/test51_speed_run

go 1.22.5

require (
	github.com/openfluke/tide v0.0.0
	github.com/openfluke/welvet v0.0.0
)

require (
	github.com/openfluke/webgpu v1.0.4 // indirect
	github.com/phpdave11/gofpdf v1.4.3 // indirect
)

replace github.com/openfluke/welvet => ../../../..

replace github.com/openfluke/webgpu => ../../../../../webgpu

replace github.com/openfluke/tide => ../../../../../tide
