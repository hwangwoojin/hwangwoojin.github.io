# hwangwoojin.github.io

hwangwoojin github pages with jekyll

## Getting started

For local, use

```s
bundle install
bundle exec jekyll serve
```

## Writing post

1. Make a markdown file for post at `/_posts` directories. Name should be `yyyy-mm-dd-title.md` fromat.

2. Add front-matter.

```s
---
layout: post
title: "My Awesome Title"
date: 2021-06-17 15:15:03 +0900
description: "My Awesome Description"
categories: category1 category2 ...
---
```

3. Write your post.

## For latex

example

```
$$ \nabla_\boldsymbol{x} J(\boldsymbol{x}) $$
```

result

![](/assets/mathjax.png)

## Reference

[Jekyll](https://jekyllrb.com/)

[Mathjax](https://www.mathjax.org/)

[Moon](https://github.com/TaylanTatli/Moon)
