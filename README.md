# strava_local_heatmap.py

Python script to reproduce the Strava Global Heatmap ([www.strava.com/heatmap](https://www.strava.com/heatmap)) with local GPX files

Optimized for cycling activities :bicyclist:

## Features

* Minimal Python dependencies (`numpy`+`matplotlib`)
* Fast (parse 3x faster than `gpxpy.parse`)

## Usage

* Download your GPX files to the `gpx` folder (https://support.strava.com/hc/en-us/articles/216918437-Exporting-your-Data-and-Bulk-Export)
* Install the python dependencies from `requirements.txt`
* Run `python strava_local_heatmap.py`

### Command-line options

```
usage: strava_local_heatmap.py [-h] [--dir DIR] [--filter FILTER] [--year YEAR [YEAR ...]]
                               [--bounds BOUND BOUND BOUND BOUND] [--output OUTPUT] [--zoom ZOOM] [--sigma SIGMA]
                               [--orange] [--csv] [--timeseries] [--since-output SINCE_OUTPUT] [--n-days N_DAYS]

Generate a PNG heatmap from local Strava GPX files

options:
  -h, --help            show this help message and exit
  --dir DIR             GPX files directory (default: gpx)
  --filter FILTER       GPX files glob filter (default: *.gpx)
  --year YEAR [YEAR ...]
                        GPX files year(s) filter (default: all)
  --bounds BOUND BOUND BOUND BOUND
                        heatmap bounding box as lat_min, lat_max, lon_min, lon_max (default: -90 +90 -180 +180)
  --output OUTPUT       heatmap name (default: heatmap.png)
  --zoom ZOOM           heatmap zoom level 0-19 or -1 for auto (default: -1)
  --sigma SIGMA         heatmap Gaussian kernel sigma in pixel (default: 1)
  --orange              not a heatmap...
  --csv                 also save the heatmap data to a CSV file
  --timeseries          create a time series for every date found in GPX files
  --since-output SINCE_OUTPUT
                        output timeseries since this date (default: 1900-01-01)
  --n-days N_DAYS       output timeseries every n days (default: 1)
```

Note: `ZOOM` is OpenStreetMap zoom level (the number following `map=` in [www.openstreetmap.org/#map=](https://www.openstreetmap.org))

On the use of histogram equalization: [https://medium.com/strava-engineering/the-global-heatmap-now-6x-hotter](https://medium.com/strava-engineering/the-global-heatmap-now-6x-hotter-23fc01d301de)

## Examples

command|output
-------|------
`strava_local_heatmap.py`|![heatmap.png](images/heatmap.png)
`strava_local_heatmap.py --orange`|![orange.png](images/orange.png)
`strava_local_heatmap.py --csv`|See https://umap.openstreetmap.fr/en/map/demo-heatmap_261644 (by [@badele](https://github.com/badele))

## Projects using strava_local_heatmap.py

[JeSuisUnDesDeux](https://gitlab.com/JeSuisUnDesDeux/jesuisundesdeux/tree/master/datas/traces)

## Troubleshooting

Arch Linux: `sudo pacman -S tk` (see [here](https://github.com/remisalmon/strava-local-heatmap/pull/3#issuecomment-443541311))
