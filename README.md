# female-pop-music-lyrics
Song lyric generator in the style of famous female pop musicians

## Data Scrape and Clean Process
```
pip install lyricsgenius
export GENIUS_ACCESS_TOKEN=<your token from https://genius.com/api-clients>
python data.py                     # both phases
python data.py --scrape            # just get raw data
python data.py --process           # just re-clean/split from cached raw
```

data is saved under 
```
data/raw
```
