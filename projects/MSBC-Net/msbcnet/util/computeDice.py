import pandas as pd
import numpy as np

dice_path = '../../diceLog.csv'


def computeDice():
    ddd = pd.read_csv(dice_path, usecols=['tumorDice','rectalWallDice','TotalDice'])
    tumorDice = ddd.tumorDice.values
    rectalWallDice = ddd.rectalWallDice.values
    TotalDice = ddd.TotalDice.values

    print('tumorDiceMean = {}'.format(np.mean(tumorDice)))
    print('rectalWallDiceMean = {}'.format(np.mean(rectalWallDice)))
    print('TotalDiceMean = {}'.format(np.mean(TotalDice)))
    # return TotalDice


if __name__ == '__main__':
    computeDice()