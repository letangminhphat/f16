# Synthesize every kernel used by the 12-layer split-kernel encoder.
# This is a csynth-only flow: no XO, link, placement, or routing is run.
set script_dir [file dirname [file normalize [info script]]]
source [file join $script_dir csynth_each_kernel.tcl]
