DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo $( pwd )
echo "Compiling cython"
bash ${DIR}/varro/cython/build_fast_cram.bash
echo "Flashing Arduino"
${DIR}/varro/arduino/flash_arduino.sh ${DIR}/varro/arduino/analog-read/
