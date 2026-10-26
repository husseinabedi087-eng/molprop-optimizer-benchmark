# Amendment 1 to PREREGISTRATION_REVISION.md (execution platform)

Recorded before any revision result was inspected. A first attempt to run the jobs on a 2-core Linux cloud
machine was stopped by the authors after about 30 minutes for lack of compute; its partial J1 outputs were
deleted without being read or analysed. All revision jobs are instead run on the machine of the original study
(Intel Core i7-10750H, Windows 11, 10 worker processes, same pinned package versions).

Consequences:
- The within-platform rule of the pre-registration is unchanged: every revision condition is compared with a
  reference condition run in the same job set. Because the platform is now the original one, J1 also checks
  that the refit of every selected configuration reproduces the published test score exactly
  (field refit_reproduces_primary_exactly), and that the pre-refit validation score reproduces the logged one.
- Code changes made before the runs, none affecting any result: J2 caches the inner folds per process; J6
  builds its descriptor cache once in the main process; default worker count 10; J1 records the two
  reproduction flags. Hypotheses, conditions, margins, decision rules and analyses are unchanged.
