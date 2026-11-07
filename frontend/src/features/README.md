# Feature modules

Each subdirectory here owns one vertical slice of the product, for example:

```
features/
  researcher/
  fact-checker/
  scriptwriter/
```

A feature owns its components, hooks and API calls, and may import from
`src/api`, `src/components`, `src/hooks` and `src/types`. Features should
not import from each other's internals; shared code belongs in one of the
shared directories.

Agent features added in later phases will depend on the backend's
`AIProvider` abstraction rather than on any specific provider, so a
different model backend can be substituted without changing them.