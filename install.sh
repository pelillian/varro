DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo $( pwd )
echo "Compiling cython"
bash ${DIR}/varro/cython/build_fast_cram.bash
echo "Flashing Arduino"


cd ${DIR}/varro/arduino/fpga-communication 
arduino-cli compile --fqbn arduino:sam:arduino_due_x_dbg
arduino-cli upload -p /dev/ttyACM0 --fqbn arduino:sam:arduino_due_x_dbg
