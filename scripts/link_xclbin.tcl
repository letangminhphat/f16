set script_dir [file dirname [file normalize [info script]]]
set root_dir [file normalize [file join $script_dir ..]]
if {![info exists ::env(U250_PLATFORM)] || $::env(U250_PLATFORM) eq ""} {
    error "Set U250_PLATFORM to the installed xilinx_u250 .xpfm path"
}
set vpp "v++"
if {[info exists ::env(VPP)] && $::env(VPP) ne ""} { set vpp $::env(VPP) }
set xo_dir [file join $root_dir build xo]
set out_dir [file join $root_dir build link]
file mkdir $out_dir
set xos {}
foreach top {
    bert_encoder_ctrl
    bert_qkv_kernel
    bert_attn_core_kernel
    bert_attn_out_norm_kernel
    bert_ffn_up_gelu_kernel
    bert_ffn_down_residual_norm_kernel
} {
    set xo [file join $xo_dir ${top}.xo]
    if {![file exists $xo]} { error "missing $xo; csynth_full_model.tcl intentionally does not export XO" }
    lappend xos $xo
}
set output [file join $out_dir bert_encoder_u250_300mhz.xclbin]
cd $root_dir
set cmd [list $vpp -l -t hw --platform $::env(U250_PLATFORM) \
    --config [file join $root_dir config system.cfg] \
    --save-temps --report_level 2 \
    --vivado.prop run.impl_1.strategy=Performance_ExplorePostRoutePhysOpt \
    -o $output]
set cmd [concat $cmd $xos]
puts "Running: $cmd"
if {[catch {exec {*}$cmd 2>@1} log]} {
    puts stderr $log
    exit 1
}
puts $log
puts "Created $output"
