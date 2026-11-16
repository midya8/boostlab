'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const {waitForDevToolsPort}=require('./chrome_support.js');
function clock() {
 let time=0,calls=0;
 return {now:()=>time,sleep:async ms=>{time+=ms;calls++;},elapsed:()=>time,calls:()=>calls};
}
test('port readiness waits for absence, partial line and complete valid port',async()=>{
 const c=clock(),states=[null,'123','12345\n/devtools/browser/id'];let i=0;
 const port=await waitForDevToolsPort('unused',{...c,timeoutMs:100,pollMs:10,readFile:async()=>{
  const value=states[i++];if(value===null)throw Object.assign(Error('absent'),{code:'ENOENT'});return value;
 }});
 assert.equal(port,12345);assert.equal(i,3);assert.equal(c.elapsed(),20);
 assert.equal(await waitForDevToolsPort('unused',{readFile:async()=> '65535\r\n/id'}),65535);
});
test('port readiness rejects invalid ports and non-absence read errors',async()=>{
 for(const value of ['0\n','65536\n'])await assert.rejects(waitForDevToolsPort('unused',{readFile:async()=>value}),/invalid debugger port/);
 await assert.rejects(waitForDevToolsPort('unused',{readFile:async()=>{throw Object.assign(Error('denied'),{code:'EACCES'});}}),/denied/);
});
test('port readiness stops at deadline and browser exit without restarting',async()=>{
 const c=clock();let reads=0;
 await assert.rejects(waitForDevToolsPort('unused',{...c,timeoutMs:30,pollMs:10,readFile:async()=>{reads++;return 'partial';}}),/deadline exceeded/);
 assert.equal(c.elapsed(),30);assert.equal(reads,3);assert.equal(c.calls(),3);
 let alive=true;const d=clock();
 await assert.rejects(waitForDevToolsPort('unused',{...d,alive:()=>alive,readFile:async()=>{alive=false;return '';}}),/exited before/);
 assert.equal(d.calls(),1);
});
