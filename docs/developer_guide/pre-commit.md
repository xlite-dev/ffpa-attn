# Prepare before commit

## 👨‍💻 Run Pre-commit

Before submitting code, configure <span style="color:#c77dff;">pre-commit</span>, for example:

```bash
# fork vipshop/ffpa-attn to your own github page, then:
git clone git@github.com:your-github-page/your-fork-ffpa-attn.git
cd your-fork-ffpa-attn && git checkout -b dev
# update submodule
git submodule update --init --recursive --force
# install pre-commit
pip3 install pre-commit
pre-commit install
pre-commit run --all-files
```

## 👨‍💻 Add a new feature

```bash
# feat: support xxx feature
# add your commits
git add .
git commit -m "support xxx-feature"
git push
# then, open a PR from your personal branch to ffpa-attn:main
```

## 👨‍💻 Check MKDocs

Please also check the <span style="color:#c77dff;">mkdocs</span> build status on your local branch.
```bash
pip3 install -e ".[docs]"
mkdocs build --strict
mkdocs serve # Then check the docs
```

Ensure that your new commits do not break the mkdocs build process.

```bash
INFO    -  Cleaning site directory
INFO    -  Building documentation to directory: /workspace/dev/ffpa-attn/site
INFO    -  Documentation built in 0.97 seconds
```
