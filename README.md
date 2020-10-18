# Fantasy Football meets Statistics

The project combines the fun of playing Fantasy Football with a analytics and prediction modelling for football data.

The goal is to support each Fantasy Football coach, by helping them to select the best 11 players for the following round line-up.

The ranking is done using the expected Fantasy Football points that would be earned the following match.
This is modelled using a compbination of player-specific and team-specific data.

The repository is strctured in 4 folders:
- Sourcing : set of functions used to download and scrape data from various sources, save it to disk, and handle specific data structure choices
- Manipulating : set of functions to combine the downloaded data into master files, team data files, player data files
- Utils : set of functions used across the board, sourced directly from other scripts
- Modelling : set of functions to create the data features and targets, train the model, predict, analyse the outcome
