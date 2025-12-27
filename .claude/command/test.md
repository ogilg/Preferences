Please test $ARGUMENT with the following procedure:
1. Read the code and understand it.
2. Write a small number of unit tests which test multiple things in the same test.
3. Run them, if they fail it could be because they were poorly designed, in that case fix them. Otherwise stop here and tell me what fails.
4. Once they pass, move onto implementing basic integration tests, make sure you test the spirit of the code and don't just write tests that fit the idiosyncrasies.
5. Run the integration tests, again you might have to iterate to catch small bugs in the tests.

Guidelines:
- NEVER implement tests that pass vacuously.