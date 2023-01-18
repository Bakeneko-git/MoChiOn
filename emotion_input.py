# This file is used to get the emotion from the user

def set_emo(num):
    convert_dict = {
                0:"neutral",
                1:"happy",
                2:"sad",
                3:"angry",
                4:"fear",
                5:"disgust",
                6:"surprise"
                }

    return convert_dict[num]
