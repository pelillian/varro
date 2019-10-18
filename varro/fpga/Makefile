PROJ=evolve
TRELLIS=~/sft/share/trellis

${PROJ}.svf : ${PROJ}.bit

prog: ${PROJ}.svf
	openocd -f ${TRELLIS}/misc/openocd/ecp5-evn.cfg -c "transport select jtag; init; svt $<; exit"

