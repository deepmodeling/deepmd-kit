# DeePMD-kit Contributing Guide

Welcome to [DeePMD-kit](https://github.com/deepmodeling/deepmd-kit)!

## What you can contribute

You can either make a code contribution, help improve our document or offer help to other users. Your help is always appreciated. Come and have fun!

### Code contribution
You can start from any one of the following items to help improve deepmd-kit

- Smash a bug
- Implement a feature or add a patch, whatever you think deepmd-kit is missing
- Browse [issues](https://github.com/deepmodeling/deepmd-kit/issues), find an issue labeled enhancement or bug, and help to solve it.

See [here](#before-you-contribute) for some before-hand heads-up.

See [here](#how-to-contribute) to learn how to contribute.

### Document improvement
You can start from any one of the following items to help improve [DeePMD-kit Docs](https://deepmd.readthedocs.io/en/latest/?badge=latest):

- Fix typos or format (punctuation, space, indentation, code block, etc.)
- Fix or update inappropriate or outdated descriptions
- Add missing content (sentence, paragraph, or a new document)
- Translate docs changes from English to Chinese

### Offer help
You can help other users of deepmd-kit in the following way

- Submit, reply to, and resolve [issues](https://github.com/deepmodeling/deepmd-kit/issues)
- (Advanced) Review Pull Requests created by others

## Before you contribute
### Overview of DeePMD-kit
Currently, we maintain two main branch:
- master: stable branch with version tag
- devel :  branch for developers

### Developer guide
See [documentation](https://deepmd.readthedocs.io/) for coding conventions, API and other needs-to-know of the code.

## How to contribute
Please perform the following steps to create your Pull Request to this repository. If don't like to use commands, you can also use [GitHub Desktop](https://desktop.github.com/), which is easier to get started. Go to [git documentation](https://git-scm.com/doc) if you want to really master git.

### Step 1: Fork the repository

1. Visit the project: <https://github.com/deepmodeling/deepmd-kit>
2. Click the **Fork** button on the top right and wait it to finish.

### Step 2: Clone the forked repository to local storage and set configurations

1. Clone your own repo, not the public repo (from deepmodeling) ! And change the branch to devel.
    ```bash
    git clone https://github.com/$username/deepmd-kit.git
    # Replace `$username` with your GitHub ID

    git checkout devel
    ```

2. Add deepmodeling's repo as your remote repo, we can name it "upstream". And fetch upstream's latest codes to your workstation.
    ```bash
    git remote add upstream https://github.com/deepmodeling/deepmd-kit.git
    # After you add a remote repo, your local repo will be automatically named "origin".

    git fetch upstream

    # If your current codes are behind the latest codes, you should merge latest codes first.
    # Notice you should merge from "devel"!
    git merge upstream/devel
    ```

3. Modify your codes and design unit tests.

4. Commit your changes
    ```bash
    git status # Checks the local status
    git add <file> ... # Adds the file(s) you want to commit. If you want to commit all changes, you can directly use `git add.`
    git commit -m "commit-message: update the xx"
    ```

5. Push the changed codes to your original repo on github.
    ```bash
    git push origin devel
    ```

### Alternatively: Create a new branch

1. Get your local master up-to-date with upstream/master.

    ```bash
    cd $working_dir/deepmd-kit
    git fetch upstream
    git checkout master
    git rebase upstream/master
    ```

2. Create a new branch based on the master branch.

    ```bash
    git checkout -b new-branch-name
    ```

3. Modify your codes and design unit tests.

4. Commit your changes

    ```bash
    git status # Checks the local status
    git add <file> ... # Adds the file(s) you want to commit. If you want to commit all changes, you can directly use `git add.`
    git commit -m "commit-message: update the xx"
    ```

5. Keep your branch in sync with upstream/master

    ```bash
    # While on your new branch
    git fetch upstream
    git rebase upstream/master
    ```

6. Push your changes to the remote

    ```bash
    git push -u origin new-branch-name # "-u" is used to track the remote branch from origin
    ```

### Step 3: Create a pull request

1. Visit your fork at <https://github.com/$username/deepmd-kit> (replace `$username` with your GitHub ID)
2. Click `pull requests`, followed by `New pull request` and `Compare & pull request` to create your PR.

Now, your PR is successfully submitted! After this PR is merged, you will automatically become a contributor to DeePMD-kit.

## Contact us
E-mail: contact@deepmodeling.org
