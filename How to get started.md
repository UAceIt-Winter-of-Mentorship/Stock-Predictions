# How to get started

<h4>To start contributing, you will need to follow the below steps in proper order:</h4>

**0.**  Create your own issue or choose one already mentioned in under issues section.

**1.** Fork [this](https://github.com/UAceIt-Winter-of-Mentorship/Stock-Predictions) repo.

**2.** Clone your fork copy of the project which will be visible in your account.

```
git clone  https://github.com/<your username>/Stock-Predictions
```

**3.** Add a remote upstream to the original repository.

```
git remote add upstream https://github.com/UAceIt-Winter-of-Mentorship/Stock-Predictions
```

**4.** Check the remotes for the repository.

```
git remote -v
```

**5.** It is always advised to take a pull from the upstream repository to your master branch to keep it even with the main project (updated repository).

```
git pull upstream master(or main)
```

**6.** Create a new branch.

```
git checkout -b <your_branch_name>
```

**7.** Perform your desired changes to the code base.

<p align="center"><img alt="Changes" width=35% src="https://media.giphy.com/media/oMHPlvpTvnXGPS7GhX/giphy.gif"></p>

**8.** Track your changes. :heavy_check_mark:

```
git add . 
```

**9.** Commit your changes.

```
git commit -m "Message related to changes you made in the code"
```

**10.** Push the committed changes in your feature branch to your remote repo.

```
git push -u origin <your_branch_name>
```

**11.** To create a pull request, click on `compare and pull requests`. Please ensure that both the branches are even in order to avoid merge conflicts.

**12.** Add a title and description to your PR explaining the features you added.

**13.** Click on `Create Pull Request`.

**14.** Congrats !! You just made a PR !! 🥳.

<p align="center"><img alt="Hooray" width=35% src="https://media.giphy.com/media/TdfyKrN7HGTIY/giphy.gif"></p>

