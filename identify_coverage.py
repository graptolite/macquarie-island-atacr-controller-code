#!/usr/bin/env python3

'''
Identify time ranges over which each station's channels recorded.

To be run after downloading (e.g. from AUSPASS) or obtaining mseed data into the structure:
  <parent dir>/<year>/<3 figure julday>/<year and 3 figure julday>_<time>_<sta>_<channel>.mseed
e.g. <parent dir>/2020/366/2020366_000000_MRO27_HDH.mseed
where parent dir is hardcoded to be "AUSPASS-MSEEDS".
'''

'''
ATaCR controller code | wrapper scripts for ATaCR as implemented by OBSTools
    Copyright (C) 2026 Yingbo Li

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import obspy
import json
import os
from tqdm import tqdm

# Years to check through.
years = ["2020","2021"]

# Where the MSEEDS downloaded from AUSPASS are stored.
parent_dir = "AUSPASS-MSEEDS"

# Iterate through last letters of the channels (must be unique).
for channel in ["Z","H","1","2"]:
    # Dictionary to store time ranges
    cha_ranges = dict()
    # Iterate through years.
    for y in years:
        year_dir_in = os.path.join(parent_dir,y)
        if os.path.exists(year_dir_in):
            # Identify (jul)days and their paths.
            days = sorted(os.listdir(year_dir_in))
            day_dirs_in = [os.path.join(year_dir_in,day) for day in days]
            # Iterate through days.
            for day_dir in tqdm(day_dirs_in):
                # Identify mseeds.
                mseeds = os.listdir(day_dir)
                # Construct key to store channel recording ranges into.
                k = y + "-" + os.path.basename(day_dir)
                cha_ranges[k] = dict()
                proc_stas = []
                # Iterate through mseeds.
                for mseed in mseeds:
                    # Identify last letter of channel code.
                    cha = mseed.split("_")[-1].split(".")[0][-1]
                    # Identify station name.
                    sta = mseed.split("_")[-2]
                    # Execute processing if the active channel is the same as the requested channel.
                    if cha == channel and sta not in proc_stas:
                        cha_ranges[k][sta] = list()
                        # Read in all mseed recordings of the day for the channel of interest as a collection of traces.
                        st = obspy.read(os.path.join(day_dir,"*_%s_*%s.mseed" % (sta,channel)))
                        # Mark the station as being processed (in case there are multiple mseed segments for the channel of interest, which would all be listed in mseeds separately but are read in by osbpy together).
                        proc_stas.append(sta)
                        # Iterate through all traces and store the trace start and end times.
                        for tr in st:
                            cha_ranges[k][sta].append((str(tr.stats.starttime),
                                                       str(tr.stats.endtime)))
    # Save channel recording ranges.
    with open("%s-covered.json" % channel,"w") as outfile:
        json.dump(cha_ranges,outfile)
