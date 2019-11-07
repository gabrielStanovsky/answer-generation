## Welcome Sameer!

Here's the starting point for adding Javascript. 
Each task `<div>` is identified by which number task it is, e.g. task1, task2, etc. 

I have three tasks, the first two of which have something to label, the third of which is empty (as denoted by the question, reference, and candidate have a value of "none"). Also the tasks which need labeling are in a variable `tasks`.

Each `<div>` task block at the end has a next and previous button. The buttons also have the id appended to the end. 

A couple of things I'm not sure about or that maybe we should clarify?

* The JavaScript code will be added directly to the HTML page right?

* If the task isn't being used, is it possible to add in Javascript an option to fill in one of the scores?

* Right now, I've hardcoded the tasks that are valid (ie that have something to label). I think it might be best to have JavaScript dynamically generate this list by seeing if the question, reference, and context are "none". 