#source $(conda info --base)/etc/profile.d/conda.sh
#conda activate base

python3 setup.py build_ext --inplace
python3 setup.py install
echo "import varro.cython.fast_cram" | python3
