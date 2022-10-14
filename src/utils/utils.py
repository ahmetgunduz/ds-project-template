# define a class map
class2idx = {
    "Joe": 0,  # user_id = 0
    "Not Joe": 1,  # user_id !=0
}
id2class = {v: k for k, v in class2idx.items()}
