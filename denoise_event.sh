
#####
# ^ dynamic params t0 and t1 above this line

# DO NOT MODIFY ANYTHING ABOVE THIS LINE.

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
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
## ^^^ ##

sta=$1

# Will read the dynamic params from above.
time_range="--start=${t0} --end=${t1}"
echo $time_range

echo "DENOISING"

# Compute spectra.
atacr_clean_spectra -O $time_range --keys=$sta AUSPASS_dataless.pkl

# Compute transfer functions.
atacr_transfer_functions -O $time_range --skip-daily --keys=$sta AUSPASS_dataless.pkl

# Execute denoising.
atacr_correct_event -O $time_range --keys=$sta AUSPASS_dataless.pkl
