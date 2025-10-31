"""
criterion
"""
import math

def get_criterion_function(criterion):
    if criterion == "info_gain":
        return __info_gain
    elif criterion == "info_gain_ratio":
        return __info_gain_ratio
    elif criterion == "gini":
        return __gini_index
    elif criterion == "error_rate":
        return __error_rate


def __label_stat(y, l_y, r_y):
    """Count the number of labels of nodes"""
    left_labels = {}
    right_labels = {}
    all_labels = {}
    for t in y.reshape(-1):
        if t not in all_labels:
            all_labels[t] = 0
        all_labels[t] += 1
    for t in l_y.reshape(-1):
        if t not in left_labels:
            left_labels[t] = 0
        left_labels[t] += 1
    for t in r_y.reshape(-1):
        if t not in right_labels:
            right_labels[t] = 0
        right_labels[t] += 1

    return all_labels, left_labels, right_labels


def __info_gain(y, l_y, r_y):
    """
    Calculate the info gain
    y, l_y, r_y: label array of father node, left child node, right child node
    """
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)
    info_gain = 0.0
    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the info gain if splitting y into      #
    # l_y and r_y                                                             #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    # Calculate entropy for parent node
    n_total = len(y)
    entropy_parent = 0.0
    for label, count in all_labels.items():
        p = count / n_total
        if p > 0:
            entropy_parent -= p * math.log2(p)
    # Calculate weighted entropy for child nodes
    n_left = len(l_y)
    n_right = len(r_y)
    entropy_left = 0.0
    for label, count in left_labels.items():
        p = count / n_left
        if p > 0:
            entropy_left -= p * math.log2(p)
    entropy_right = 0.0
    for label, count in right_labels.items():
        p = count / n_right
        if p > 0:
            entropy_right -= p * math.log2(p)
    # Info gain = parent entropy - weighted child entropy
    info_gain = entropy_parent - (n_left / n_total * entropy_left + n_right / n_total * entropy_right)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return info_gain


def __info_gain_ratio(y, l_y, r_y):
    """
    Calculate the info gain ratio

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    info_gain = __info_gain(y, l_y, r_y)
    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the info gain ratio if splitting y     #
    # into l_y and r_y                                                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Calculate split information (intrinsic value)
    n_total = len(y)
    n_left = len(l_y)
    n_right = len(r_y)

    split_info = 0.0
    if n_left > 0:
        p_left = n_left / n_total
        split_info -= p_left * math.log2(p_left)
    if n_right > 0:
        p_right = n_right / n_total
        split_info -= p_right * math.log2(p_right)

    # Info gain ratio = info gain / split info
    if split_info > 0:
        info_gain_ratio = info_gain / split_info
    else:
        info_gain_ratio = 0.0

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return info_gain_ratio


def __gini_index(y, l_y, r_y):
    """
    Calculate the gini index

    y, l_y, r_y: label array of father node, left child node, right child node
    """
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)
    before = 0.0
    after = 0.0

    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the gini index value before and        #
    # after splitting y into l_y and r_y                                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Calculate Gini index before split
    n_total = len(y)
    before = 1.0
    for label, count in all_labels.items():
        p = count / n_total
        before -= p ** 2

    # Calculate Gini index after split (weighted)
    n_left = len(l_y)
    n_right = len(r_y)

    gini_left = 1.0
    for label, count in left_labels.items():
        p = count / n_left
        gini_left -= p ** 2

    gini_right = 1.0
    for label, count in right_labels.items():
        p = count / n_right
        gini_right -= p ** 2

    after = (n_left / n_total) * gini_left + (n_right / n_total) * gini_right

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return before - after


def __error_rate(y, l_y, r_y):
    """Calculate the error rate"""
    all_labels, left_labels, right_labels = __label_stat(y, l_y, r_y)
    before = 0.0
    after = 0.0

    ###########################################################################
    # TODO:                                                                   #
    # Implement this method. Calculate the error rate value before and        #
    # after splitting y into l_y and r_y                                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Calculate error rate before split
    # Error rate = 1 - max_probability
    n_total = len(y)
    max_count = max(all_labels.values())
    before = 1.0 - (max_count / n_total)

    # Calculate error rate after split (weighted)
    n_left = len(l_y)
    n_right = len(r_y)

    max_count_left = max(left_labels.values()) if left_labels else 0
    error_left = 1.0 - (max_count_left / n_left) if n_left > 0 else 0

    max_count_right = max(right_labels.values()) if right_labels else 0
    error_right = 1.0 - (max_count_right / n_right) if n_right > 0 else 0

    after = (n_left / n_total) * error_left + (n_right / n_total) * error_right

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return before - after
