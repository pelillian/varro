#source $(conda info --base)/etc/profile.d/conda.sh
#conda activate base
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd ${DIR}
python3 setup.py build_ext --inplace
python3 setup.py install

LIB=${DIR}/build/lib.linux-x86_64-3.7/varro/cython/fast_cram.cpython-37m-x86_64-linux-gnu.so
LINK_LOCATION=${DIR}/$(basename $LIB)
if [ ! -f ${LINK_LOCATION} ]; then
    ln -s ${LIB} ${LINK_LOCATION}
fi

echo "import varro.cython.fast_cram" | python3
