# Code-only HLS synthesis of each device-managed kernel.  This script does not
# export XO, run Vivado logic synthesis, link, place, or route.
#
# Run all six kernels:
#   vitis_hls -f scripts/csynth_each_kernel.tcl
# Select a subset:
#   BERT_KERNELS=bert_encoder_ctrl,bert_qkv_kernel vitis_hls -f scripts/csynth_each_kernel.tcl

set script_dir [file dirname [file normalize [info script]]]
set root_dir [file normalize [file join $script_dir ..]]
set code_dir [file join $root_dir code]
set build_dir [file join $root_dir build csynth_each]
set report_dir [file join $root_dir report csynth_current]
set detail_root [file join $report_dir details]

set part_name "xcu250-figd2104-2L-e"
set clock_mhz 300.0
if {[info exists ::env(BERT_CLOCK_MHZ)]} {
    if {![string is double -strict $::env(BERT_CLOCK_MHZ)] ||
        $::env(BERT_CLOCK_MHZ) <= 0.0} {
        error "BERT_CLOCK_MHZ must be a positive number"
    }
    set clock_mhz [expr {double($::env(BERT_CLOCK_MHZ))}]
}
set clock_ns [expr {1000.0 / $clock_mhz}]

array set source_for {
    bert_encoder_ctrl bert_encoder_ctrl.cpp
    bert_qkv_kernel bert_qkv_kernel.cpp
    bert_attn_core_kernel bert_attn_core_kernel.cpp
    bert_attn_out_norm_kernel bert_attn_out_norm_kernel.cpp
    bert_ffn_up_gelu_kernel bert_ffn_kernel_v21.cpp
    bert_ffn_down_residual_norm_kernel bert_ffn_kernel_v21.cpp
}
set all_kernels {
    bert_encoder_ctrl
    bert_qkv_kernel
    bert_attn_core_kernel
    bert_attn_out_norm_kernel
    bert_ffn_up_gelu_kernel
    bert_ffn_down_residual_norm_kernel
}

file mkdir $build_dir
file mkdir $report_dir
file mkdir $detail_root

proc show_failure_logs {project_dir} {
    set db [file join $project_dir solution1 .autopilot db]
    foreach pattern [list "*clang*.err.log" "*.opt.err.log" "*.llvm-link.err.log"] {
        foreach path [glob -nocomplain [file join $db $pattern]] {
            if {[file isfile $path] && [file size $path] > 0} {
                puts stderr "---- $path ----"
                set fp [open $path r]
                puts stderr [read $fp]
                close $fp
            }
        }
    }
}

proc preflight {top source_path} {
    if {![file isfile $source_path]} { error "missing source: $source_path" }
    set fp [open $source_path r]
    set text [read $fp]
    close $fp
    if {![regexp -- "void\\s+$top\\s*\\(" $text]} {
        error "top function '$top' is not declared in $source_path"
    }
}

proc copy_reports {project_dir top} {
    global report_dir detail_root
    set source_dir [file join $project_dir solution1 syn report]
    set primary [file join $source_dir csynth.rpt]
    if {![file isfile $primary]} {
        set primary [file join $source_dir ${top}_csynth.rpt]
    }
    if {![file isfile $primary] || [file size $primary] == 0} {
        error "missing csynth report for $top in $source_dir"
    }
    file copy -force $primary [file join $report_dir csynth_${top}.rpt]

    set detail_dir [file join $detail_root $top]
    file delete -force $detail_dir
    file mkdir $detail_dir
    foreach path [glob -nocomplain [file join $source_dir *]] {
        if {[file isfile $path] && [file size $path] > 0} {
            file copy -force $path $detail_dir
        }
    }
}

proc synth_one {top source_name} {
    global root_dir code_dir build_dir part_name clock_mhz clock_ns
    set source_path [file join $code_dir $source_name]
    preflight $top $source_path
    set project_name proj_${top}
    set project_dir [file join $build_dir $project_name]
    set old_dir [pwd]

    puts "============================================================"
    puts "CSYNTH ONLY: $top"
    puts "Source: $source_path"
    puts "Clock : $clock_mhz MHz ($clock_ns ns)"
    puts "============================================================"

    cd $build_dir
    open_project -reset $project_name
    add_files -cflags "-std=c++14 -I$code_dir" $source_path
    set_top $top
    open_solution -reset solution1 -flow_target vivado
    set_part $part_name
    create_clock -period $clock_ns -name default
    csynth_design
    copy_reports $project_dir $top
    close_project
    cd $old_dir
}

set selected $all_kernels
if {[info exists ::env(BERT_KERNELS)] && $::env(BERT_KERNELS) ne ""} {
    set selected {}
    foreach requested [split $::env(BERT_KERNELS) ","] {
        set top [string trim $requested]
        if {![info exists source_for($top)]} {
            error "unknown kernel in BERT_KERNELS: $top"
        }
        if {[lsearch -exact $selected $top] >= 0} {
            error "duplicate kernel in BERT_KERNELS: $top"
        }
        lappend selected $top
    }
}

set failed {}
foreach top $selected {
    set project_dir [file join $build_dir proj_${top}]
    if {[catch {synth_one $top $source_for($top)} message]} {
        puts stderr "FAIL $top: $message"
        show_failure_logs $project_dir
        catch {close_project}
        catch {cd $script_dir}
        lappend failed $top
    } else {
        puts "PASS $top"
    }
}

if {[llength $failed] > 0} {
    puts stderr "FAILED: [join $failed {, }]"
    exit 1
}
puts "PASS: all selected kernels completed csynth."
puts "Reports: $report_dir"
exit 0
