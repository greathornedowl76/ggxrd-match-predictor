# ggxrd-match-predictor
Parses through Guilty Gear Xrd Rev 2 matches to collect data for a match predictor. 

# Overview
The primary goal of this project is to ultimately collect enough data from match videos in order to predict the outcome of a given Guilty Gear Xrd Rev 2 match. Currently, this project is deep within the data collection phase. It aims to consider not just surface level traits such as character's played, and player's rank, but also a player's neutral interactions such as direct-attack/counter-attack/second-intention success rates, kd game, and defensive traits. 

| ![Boxes on States](https://raw.githubusercontent.com/ravenseattuna/ggxrd-match-predictor/master/screenshots/compiled.png) |
|:--:| 
| *From top left to bottom right: attack-lunge, attack-stand, attack-crouch, attack-air, knockdown, advance(run), retreat, neutral, and advance(air dash)* |
# How It Will Work
Currently uses DarkFlow to detect a character's in-game state (i.e. attacking, advaning, retreating, etc.). Will be used to create a finite state automoton to detect more advanced neutral states such as counter-attacks and direct-attacks (oki-waza and ate-waza respectively). 

Hits are detected through the appearearance of red health in a player's health bar, and the start of a match is detected through the appearahce of "99" on the timer. Takes large influence from nxth's and keeponrockin's ggxrd parsers. 

# Changelog
11/4/19: Uploaded to Github. Added hit detection by looking for red health during match start, duel starts by detecting "99" in the timer, and victories by checking if the victory bulbs below each character's health bar is on or off. Added ability to OCR rank of a player, bugfixes to OCR name detection.



