#!/bin/bash
for f in *.tex
do
awk '
/begin.notebox/
{print $0,"\n\\textbf\{Note\}"} 

/begin.tipbox/
{print $0,"\n\\textbf\{Tip\}"} 

/begin.stepbox/
{print $0,"\n\\textbf\{Step-by-step\}"} 

/begin.warnbox/
{print $0,"\n\\textbf\{Tip\}"} 

!/begin\{[a-z]+box\}/
{print $0}
' $f > $f
done
