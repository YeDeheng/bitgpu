set TOP [file rootname [info script]]
delete_project	${TOP}_batch.prj
open_project	${TOP}_batch.prj
add_files 		${TOP}.cpp
set_top  		${TOP}
open_solution	solution1
set_part 		xc7k160tfbg484-2
create_clock	-period 3ns
csynth_design
export_design
exit
