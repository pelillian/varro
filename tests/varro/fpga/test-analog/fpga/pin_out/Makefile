PROJ=pin_out
TRELLIS=~/sft/share/trellis

all: ${PROJ}.bit

%.json: %.v
	yosys -p "synth_ecp5 -json $@" $<

%_out.config: %.json
	nextpnr-ecp5 --json $< --textcfg $@ --um5g-85k --package CABGA381 --lpf ecp5evn.lpf

%.bit: %_out.config
	ecppack --svf ${PROJ}.svf $< $@ 

${PROJ}.svf: ${PROJ}.bit

bash_script: SHELL:=/bin/bash 

bash_script:
	bash -c "more pin_out_out.config | grep .tile > tiles.txt"

prog: ${PROJ}.svf bash_script
	openocd -f ${TRELLIS}/misc/openocd/ecp5-evn.cfg -c "transport select jtag; init; svf $<; exit" ;\

clean:
	rm -f *.svf *.bit *.config *.json *.txt

PHONY: prog clean bash_script

.SECONDARY:

