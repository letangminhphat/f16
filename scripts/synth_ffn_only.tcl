# Synthesize only the two FFN kernels.
# Run from any directory with:
#   vitis_hls -f scripts/synth_ffn_only.tcl

set script_dir [file dirname [file normalize [info script]]]
set root_dir   [file normalize [file join $script_dir ..]]
set kernel_dir [file join $root_dir code]
set include_dir [file join $root_dir code]
set build_dir  [file join $root_dir build hls_ffn]
set report_dir [file join $root_dir report ffn]

set part_name "xcu250-figd2104-2L-e"
set clock_mhz 300.0
if {[info exists ::env(BERT_CLOCK_MHZ)]} {
    set clock_mhz [expr {double($::env(BERT_CLOCK_MHZ))}]
}
if {$clock_mhz <= 0.0} {
    error "BERT_CLOCK_MHZ must be positive"
}
set clock_ns [expr {1000.0 / $clock_mhz}]

set ffn_kernels {
    {bert_ffn_up_gelu_kernel bert_ffn_kernel_v21.cpp}
    {bert_ffn_down_residual_norm_kernel bert_ffn_kernel_v21.cpp}
}

file mkdir $build_dir
file mkdir $report_dir

proc preflight_source {top source_path} {
    global include_dir

    if {![file isfile $source_path]} {
        error "source file not found: $source_path"
    }

    set fp [open $source_path r]
    set source_text [read $fp]
    close $fp

    if {![regexp -- "void\\s+$top\\s*\\(" $source_text]} {
        error "top function '$top' is not declared in $source_path"
    }

    set protocol_header [file join $include_dir bert_encoder_protocol.h]
    set ffn_header [file join $include_dir bert_ffn_kernel_lab.h]
    if {![file isfile $protocol_header]} {
        error "required header not found: $protocol_header"
    }
    if {![file isfile $ffn_header]} {
        error "required header not found: $ffn_header"
    }
}

proc synth_ffn_kernel {top source_name} {
    global kernel_dir include_dir build_dir report_dir
    global part_name clock_ns

    set source_path [file join $kernel_dir $source_name]
    preflight_source $top $source_path

    set project_name "proj_${top}"
    set project_dir [file join $build_dir $project_name]
    set old_dir [pwd]
    cd $build_dir

    puts "============================================================"
    puts "Synthesizing $top"
    puts "Source : $source_path"
    puts "Project: $project_dir"
    puts "============================================================"

    open_project -reset $project_name
    set_top $top
    add_files -cflags "-std=c++14 -I$include_dir" $source_path

    open_solution -reset solution1 -flow_target vitis
    set_part $part_name
    create_clock -period $clock_ns -name default
    csynth_design

    set report_src [file join $project_dir solution1 syn report csynth.rpt]
    if {![file isfile $report_src]} {
        set report_src [file join $project_dir solution1 syn report ${top}_csynth.rpt]
    }
    if {![file isfile $report_src]} {
        error "csynth completed but report was not found for $top"
    }
    file copy -force $report_src [file join $report_dir ${top}_csynth.rpt]

    close_project
    cd $old_dir
}

set failed {}
foreach spec $ffn_kernels {
    lassign $spec top source_name
    if {[catch {synth_ffn_kernel $top $source_name} message options]} {
        puts stderr "FAIL $top: $message"
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

puts "PASS: both FFN kernels synthesized."
puts "Reports: $report_dir"
exit 0
