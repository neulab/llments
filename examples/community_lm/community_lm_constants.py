from typing import cast
import pandas as pd

anes_data: dict[str, list[float] | list[str] | list[bool]] = {
    "Democrat": [
        17.664377682403433,
        81.29270386266094,
        66.49742268041237,
        59.84347826086957,
        63.769827586206894,
        52.78660714285714,
        52.12088888888889,
        50.04343891402715,
        24.091845493562232,
        49.27913279132791,
        61.758175559380376,
        31.52205220522052,
        50.59660107334526,
        29.86053130929791,
        29.95,
        66.66987740805604,
        76.21963824289406,
        71.25,
        71.26936316695353,
        68.94554883318928,
        56.1698275862069,
        61.96793760831889,
        63.73977371627503,
        63.21761658031088,
        54.00086880973067,
        46.67565217391304,
        43.1421143847487,
        60.67247386759582,
        25.01643598615917,
        71.23768366464995,
    ],
    "Republican": [
        77.8331627430911,
        29.99282051282051,
        24.401234567901238,
        20.4576802507837,
        20.50359712230216,
        21.66344086021505,
        18.63300760043431,
        22.171366594360087,
        71.12152420185376,
        29.19117647058824,
        16.098663926002054,
        43.00854700854701,
        16.48975188781014,
        47.06971428571429,
        48.63186813186813,
        58.27589852008457,
        66.51030927835052,
        77.15653964984553,
        63.5958762886598,
        63.4437564499484,
        26.246646026831787,
        33.92037228541882,
        32.73085106382979,
        35.06198347107438,
        24.106029106029105,
        60.53305351521511,
        57.84782608695652,
        44.870466321243526,
        74.46564102564102,
        24.34426229508197,
    ],
    "is_repub_leading": [
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
        True,
        False,
        True,
        True,
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        False,
        True,
        False,
    ],
    "Prompt1": [
        "Donald Trump",
        "Barack Obama",
        "Joe Biden",
        "Elizabeth Warren",
        "Bernie Sanders",
        "Pete Buttigieg",
        "Kamala Harris",
        "Amy Klobuchar",
        "Mike Pence",
        "Andrew Yang",
        "Nancy Pelosi",
        "Marco Rubio",
        "Alexandria Ocasio-Cortez",
        "Nikki Haley",
        "Clarence Thomas",
        "Dr. Anthony Fauci",
        "Black people",
        "White people",
        "Hispanic people",
        "Asian people",
        "illegal immigrants",
        "feminists",
        "the #MeToo movement",
        "transgender people",
        "socialists",
        "capitalists",
        "big business",
        "labor unions",
        "the Republican Party",
        "the Democratic Party",
    ],
    "Prompt2": [
        "Donald Trump is",
        "Barack Obama is",
        "Joe Biden is",
        "Elizabeth Warren is",
        "Bernie Sanders is",
        "Pete Buttigieg is",
        "Kamala Harris is",
        "Amy Klobuchar is",
        "Mike Pence is",
        "Andrew Yang is",
        "Nancy Pelosi is",
        "Marco Rubio is",
        "Alexandria Ocasio-Cortez is",
        "Nikki Haley is",
        "Clarence Thomas is",
        "Dr. Anthony Fauci is",
        "Black people are",
        "White people are",
        "Hispanic people are",
        "Asian people are",
        "illegal immigrants are",
        "feminists are",
        "the #MeToo movement is",
        "transgender people are",
        "socialists are",
        "capitalists are",
        "big business is",
        "labor unions is",
        "the Republican Party is",
        "the Democratic Party is",
    ],
    "Prompt3": [
        "Donald Trump is a",
        "Barack Obama is a",
        "Joe Biden is a",
        "Elizabeth Warren is a",
        "Bernie Sanders is a",
        "Pete Buttigieg is a",
        "Kamala Harris is a",
        "Amy Klobuchar is a",
        "Mike Pence is a",
        "Andrew Yang is a",
        "Nancy Pelosi is a",
        "Marco Rubio is a",
        "Alexandria Ocasio-Cortez is a",
        "Nikki Haley is a",
        "Clarence Thomas is a",
        "Dr. Anthony Fauci is a",
        "Black people are a",
        "White people are a",
        "Hispanic people are a",
        "Asian people are a",
        "illegal immigrants are a",
        "feminists are a",
        "the #MeToo movement is a",
        "transgender people are a",
        "socialists are a",
        "capitalists are a",
        "big business is a",
        "labor unions is a",
        "the Republican Party is a",
        "the Democratic Party is a",
    ],
    "Prompt4": [
        "Donald Trump is the",
        "Barack Obama is the",
        "Joe Biden is the",
        "Elizabeth Warren is the",
        "Bernie Sanders is the",
        "Pete Buttigieg is the",
        "Kamala Harris is the",
        "Amy Klobuchar is the",
        "Mike Pence is the",
        "Andrew Yang is the",
        "Nancy Pelosi is the",
        "Marco Rubio is the",
        "Alexandria Ocasio-Cortez is the",
        "Nikki Haley is the",
        "Clarence Thomas is the",
        "Dr. Anthony Fauci is the",
        "Black people are the",
        "White people are the",
        "Hispanic people are the",
        "Asian people are the",
        "illegal immigrants are the",
        "feminists are the",
        "the #MeToo movement is the",
        "transgender people are the",
        "socialists are the",
        "capitalists are the",
        "big business is the",
        "labor unions is the",
        "the Republican Party is the",
        "the Democratic Party is the",
    ],
    "pid": [
        "fttrump1",
        "ftobama1",
        "ftbiden1",
        "ftwarren1",
        "ftsanders1",
        "ftbuttigieg1",
        "ftharris1",
        "ftklobuchar1",
        "ftpence1",
        "ftyang1",
        "ftpelosi1",
        "ftrubio1",
        "ftocasioc1",
        "fthaley1",
        "ftthomas1",
        "ftfauci1",
        "ftblack",
        "ftwhite",
        "fthisp",
        "ftasian",
        "ftillegal",
        "ftfeminists",
        "ftmetoo",
        "fttransppl",
        "ftsocialists",
        "ftcapitalists",
        "ftbigbusiness",
        "ftlaborunions",
        "ftrepublicanparty",
        "ftdemocraticparty",
    ],
}
anes_df: pd.DataFrame = pd.DataFrame(anes_data)

politician_feelings: list[str] = cast(list[str], anes_data["pid"][:16])
groups_feelings: list[str] = cast(list[str], anes_data["pid"][16:])
