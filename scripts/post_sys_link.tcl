# Keep the generated AXIS register slices and SLR crossings visible to
# physical optimization.  The commands are guarded because instance names vary
# slightly between U250 platform releases.
set axis_cells [get_cells -quiet -hier -filter {NAME =~ *axis*register*slice*}]
if {[llength $axis_cells] > 0} {
    set_property DONT_TOUCH false $axis_cells
}
set_property SEVERITY Warning [get_drc_checks -quiet HDPR-30]
