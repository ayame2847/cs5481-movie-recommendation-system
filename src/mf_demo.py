"""Display unseen movie recommendations from a fitted MF checkpoint."""
import argparse
from pathlib import Path
import numpy as np
from mf import BiasedMF, load_shared, recommend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--user', type=int, help='Omit to enter user IDs interactively')
    args = parser.parse_args()
    data, model = load_shared(args.project), BiasedMF.load(args.model)
    if not np.array_equal(data.item_ids, model.item_ids):
        raise ValueError('Model catalogue and input catalogue differ')
    def display(user):
        if user not in model.user_index:
            print('New user: recommendations use the global mean and movie biases.')
        print(recommend(model, data, user, 'final', 5).to_string(index=False))
    if args.user is not None:
        display(args.user)
        return
    print('Enter a user ID, or q to quit.')
    while True:
        try:
            value = input('User ID: ').strip()
        except EOFError:
            break
        if value.lower() == 'q':
            break
        try:
            user = int(value)
        except ValueError:
            print('Please enter an integer user ID.')
            continue
        display(user)


if __name__ == '__main__':
    main()
