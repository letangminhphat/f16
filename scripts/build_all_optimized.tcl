# Compatibility entry point: csynth all six device-managed BERT encoder
# kernels.  This flow intentionally does not export XO or invoke Vivado.
set script_dir [file dirname [file normalize [info script]]]
source [file join $script_dir csynth_each_kernel.tcl]
