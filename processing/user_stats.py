import operator
from collections import defaultdict
import matplotlib.pyplot as plt

"""
Helper methods for the user statistics section of the analysis
"""

# check if user data exists before doing anything
def check_for_user_data(data):
    has_user_data = False
    count = 0
    for entry in data:
        if 'user' in entry:
            has_user_data = True
            count += 1
    return has_user_data, count / len(data)

# sort posts by user that posted them
def get_posts_per_user(data):
    user_ids = dict()
    user_posts = dict()
    for entry in data:
        user = entry['user']['id']
        if not user in user_ids:
            user_ids[user] = 1
        else:
            user_ids[user] +=1
        if not user in user_posts:
            user_posts[user] = [(entry['id'], entry['label'])]
        else:
            user_posts[user].append((entry['id'], entry['label']))

    user_hitlist = sorted(user_ids.items(), key=operator.itemgetter(1), reverse=True)
    sum = 0
    for user in user_hitlist[:4]:
        sum += user[1]
    return user_hitlist, user_ids, user_posts

# how many posts per user
def get_post_stats(user, user_posts, labels):
    l_count = defaultdict(int)
    for entry in user_posts[user]:
        l_count[entry[1]] += 1
    return l_count

# get all stats for all users
def get_user_stats(users, user_posts, labels):
    user_stats = dict()
    for user in users:
        l_count = get_post_stats(user[0], user_posts, labels)
        user_stats[user[0]] = l_count
    return user_stats

# helper method to display the top users and remainder as a plot
def show_user_dist(user_hitlist, top_n=5):
    labels = list()
    sizes = list()
    for num, user in enumerate(user_hitlist):
        if num == top_n:
            labels.append("Rest")
            sizes.append(user[1])
        elif num > top_n:
            sizes[-1] += user[1]
        else:
            labels.append(str(user[0]))
            sizes.append(user[1])

    fig1, ax1 = plt.subplots()
    #ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
    #        shadow=True, startangle=90)
    #ax1.axis('equal')
    ax1.barh(labels, sizes)
    plt.show()

# differentiates between label to check individual user distributions
def show_user_dist_by_label(user_hitlist, user_stats, label, top_n=5):
    labels = list()
    sizes = list()
    for num, user in enumerate(user_hitlist):
        if num == top_n:
            labels.append("Rest")
            sizes.append(user_stats[user[0]][label])
        elif num > top_n:
            sizes[-1] += user_stats[user[0]][label]
        else:
            labels.append(str(user[0]))
            sizes.append(user_stats[user[0]][label])

    fig1, ax1 = plt.subplots()
    #ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
    #        shadow=True, startangle=90)
    #ax1.axis('equal')
    ax1.barh(labels, sizes)
    plt.show()
