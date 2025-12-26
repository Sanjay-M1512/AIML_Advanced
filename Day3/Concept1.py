import os
import sys
import numpy as np

def entropy(probabilities):

    return -np.sum([p * np.log2(p) for p in probabilities if p > 0])


def calculate_information_gain(parent_entropy, child_splits):

    total = sum([count for count, ent in child_splits])
    weighted_entropy = sum([(count / total) * ent for count, ent in child_splits])
    return parent_entropy - weighted_entropy


def main():

    filename = input("").strip()
    file_path = os.path.join(sys.path[0], filename)

    try:
        data = np.genfromtxt(file_path, delimiter=",", skip_header=1)
    except:
        print(f"Error: Unable to read file '{filename}'.")
        sys.exit(1)

    target = data[:, -1]  

    total = len(target)
    likely = np.sum(target == 1)
    unlikely = np.sum(target == 0)

    parent_entropy = entropy([likely / total, unlikely / total])
    print(f"Parent Node Entropy: {parent_entropy:.3f}")

    fb = data[:, 0] > 0  

    yes = target[fb]
    no = target[~fb]

    fb_split = [
        (len(yes), entropy([np.sum(yes == 1) / len(yes) if len(yes)>0 else 0,
                            np.sum(yes == 0) / len(yes) if len(yes)>0 else 0])),
        (len(no), entropy([np.sum(no == 1) / len(no) if len(no)>0 else 0,
                           np.sum(no == 0) / len(no) if len(no)>0 else 0]))
    ]

    fb_ig = calculate_information_gain(parent_entropy, fb_split)
    print(f"Information Gain (Fasting blood): {fb_ig:.3f}")

    bmi = data[:, 1] > 0

    yes = target[bmi]
    no = target[~bmi]

    bmi_split = [
        (len(yes), entropy([np.sum(yes == 1)/len(yes) if len(yes)>0 else 0,
                            np.sum(yes == 0)/len(yes) if len(yes)>0 else 0])),
        (len(no), entropy([np.sum(no == 1)/len(no) if len(no)>0 else 0,
                           np.sum(no == 0)/len(no) if len(no)>0 else 0]))
    ]

    bmi_ig = calculate_information_gain(parent_entropy, bmi_split)
    print(f"Information Gain (bmi): {bmi_ig:.3f}")

    fam = data[:, 3] > 0

    yes = target[fam]
    no = target[~fam]

    fam_split = [
        (len(yes), entropy([np.sum(yes == 1)/len(yes) if len(yes)>0 else 0,
                            np.sum(yes == 0)/len(yes) if len(yes)>0 else 0])),
        (len(no), entropy([np.sum(no == 1)/len(no) if len(no)>0 else 0,
                           np.sum(no == 0)/len(no) if len(no)>0 else 0]))
    ]

    fam_ig = calculate_information_gain(parent_entropy, fam_split)
    print(f"Information Gain (FamilyHistory): {fam_ig:.3f}")

    results = {
        "Fasting blood": fb_ig,
        "bmi": bmi_ig,
        "FamilyHistory": fam_ig
    }

    best = max(results, key=results.get)
    print(f"Best Feature for root node: {best} with Information Gain: {results[best]:.3f}")


if __name__ == "__main__":
    main()