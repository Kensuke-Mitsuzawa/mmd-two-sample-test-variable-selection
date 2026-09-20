---
name: python-coding.
description: Behavioral guidelines that an agent has to follow when the agent is coding in python.
license: MIT
---

# Python Coding Guidelines

## Data object by Pydantic 

This rule is applied when a function/method returns more than two values.

You are asked to define the Pydantic data class.

First, you import `from pydantic import BaseModel, Field` at the module's header.

Then, you define the data object like the following example.

```python
class DBrecordData(BaseModel):
    """descirption
    """
    field_name: Typing = Field(description="")   
```

## Typing

This rule is applied when you define a function or method.

First, you import the `typing` module as `import typing as ty` or use the typing hints from `typing` module.

You are asked to set the type hint at the function/method argument and the return obejct.

## Naming of variable, function, method, and class.

To name a variable, function, and method, you are aked to name them in the latin language style; the token's order is "verb-noun-adjectives". A underbar is used to separate the tokens.

For the class naming, the naming rule is "Adjective (or Noun) Noun" that is the german language style. You are asked to make the best effort to name "-er, -or" sufixes to the verb form noun to represen the verb operation. For example, when you write the "execute" (verb), you have to name "Executor" class.

## Class structure

A class should be implemented in the following order,

- `__init__` if required.
- public methods comes first.
- semi-private, which starts with a single underbar "_" comes after public methods.
- private, which starts with a double underbar "__" comes after semi-private methods.


## end of if/for/while/try/with blocks

When you write a code using if/for/while/try/with, you are asked to put a comment line `# end <block name>`.
