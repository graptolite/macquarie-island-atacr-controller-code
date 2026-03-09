# Generate daily spectra and daily transfer functions once (to avoid regenerating the same things each time).

# ATaCR controller code | wrapper scripts for ATaCR as implemented by OBSTools
#     Copyright (C) 2026 Yingbo Li

#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

# To prevent numpy taking over too many threads
# After https://stackoverflow.com/a/53224849
# Posted by Amir, modified by community. See post 'Timeline' for change history
# Retrieved 2026-02-24, License - CC BY-SA 4.0
## vvv ##
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
## ^^^ ##

# List all stations
stas="MRO01 MRO03 MRO06 MRO07 MRO10 MRO11 MRO12 MRO13 MRO16 MRO18 MRO20 MRO22 MRO23 MRO24 MRO27"
network="3F"

echo "generating spectra for"
echo $stas

# Make sure that the file format is expected.
# *CHANGE THE PATH TO THE MINICONDA FOLDER AS NECESSARY*
miniconda_dir="~/miniconda3"
src="SAC"
repl="mseed"
sed -i -e "s=${src}=${repl}=g" $miniconda_dir/lib/python3.12/site-packages/obstools/atacr/utils.py

# Make sure the logs dir as available
mkdir logs

# Code for regenerating daily spectra and daily transfer functions.
proc_sta() {
local sta=$1
atacr_daily_spectra --flag-freqs=0.02,2 --tilt-freqs=0.01,0.04 --tolerance=1.5 --keys=$sta AUSPASS_dataless.pkl > logs/$sta.log 2> logs/$sta.err
atacr_transfer_functions --keys=$sta AUSPASS_dataless.pkl
}

# THIS WILL USE ONE CORE (execution in series).
# For execution in parallel where the machine has 15 (same as number of stations) cores available uncomment the &
for sta in $stas; do
proc_sta $sta # &
done
