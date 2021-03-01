# this file should be run with `source flash_arduino.sh`

if [ "$#" -ne 1 ] || ! [ -d "$1" ]; then
   echo "Usage: $0 DIRECTORY" >&2
   exit 1
fi

echo "Flashing ino in directory:"
realpath $1

cd $1
arduino-cli compile --fqbn arduino:sam:arduino_due_x_dbg
arduino-cli upload -p /dev/ttyACM0 --fqbn arduino:sam:arduino_due_x_dbg
