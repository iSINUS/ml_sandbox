import pandas as pd
import os
import git

from ...utils.zipcode import GeoDistance

package_directory = os.path.dirname(os.path.abspath(__file__))
data_dir = package_directory + '/../../data/'
log_dir = package_directory + '/../../logs/'
model_dir = package_directory + '/../../model/'


def GetRepo(work_dir):
    try:
        repo = git.Repo(work_dir)
    except git.exc.InvalidGitRepositoryError:
        repo = git.Repo.init(work_dir)
        repo.index.add('.')
        repo.index.commit('initial commit')
    print(work_dir, 'Last commit on:', repo.head.commit.committed_datetime)

    return repo


def gitCommitData(comment):
    repo = GetRepo(data_dir)
    repo.index.add('.')
    repo.index.commit(comment)


def gitCommitModel(comment):
    repo = GetRepo(model_dir)
    repo.index.add('.')
    repo.index.commit(comment)


def getDistance(origin, destination):

    return dist.query_postal_code(origin, destination)


dist = GeoDistance()