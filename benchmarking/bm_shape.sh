#!/bin/bash

if [ $# -ne 3 ];
then
	echo "$0: benchmark_cpp_file_name test_name shape"
	exit 0
fi

BM_CPP=$1
TEST=$2
SHAPE=$3

BM_PATH=../tests/gaudi2_test/${BM_CPP}

# Obtain fresh copy of benchmark test
git checkout ${BM_PATH}

# Change shape in the benchmark test
sed -i 's/{128, 3}/'"${SHAPE}"'/g' ${BM_PATH}

pushd ../build
make -j >& /dev/null

# Change shape from {x, y} to x_y so that we can use that as suffix to CSV file.
SHAPE_STR=`echo ${SHAPE} | tr -d "{} " | tr -s "," "_"`
hl-prof-config --config-filename=../metalift.json --gaudi2 --phase=device-acq -b 256 -o /home/sdp/homes/nhasabni/Habana_Custom_Kernel/profiling --invoc csv -s ${TEST}_${SHAPE_STR}

# Run the test
HABANA_PROF_CONFIG=../metalift.json TPC_RUNNER=1 HABANA_PROFILE=1 tests/tpc_kernel_tests -t ${TEST} >& /dev/null
if [ $? -ne 0 ];
then
  echo "Test execution failed"
  exit 0
fi

# Obtain start and end times.
start_time=`grep -e "TRC_TPC_LOG_CONTEXT_START.*TPC 0,BEGIN" ../profiling/${TEST}_${SHAPE_STR}_accel0.csv | awk -F "," '{print $5}'`
end_time=`grep -e "TRC_TPC_LOG_CONTEXT_HALT.*TPC 0,END" ../profiling/${TEST}_${SHAPE_STR}_accel0.csv | awk -F "," '{print $5}'`

bm_time=`echo "scale=2; $end_time - $start_time" | bc`

echo ${TEST},${SHAPE},${bm_time}"(usec)"

popd
