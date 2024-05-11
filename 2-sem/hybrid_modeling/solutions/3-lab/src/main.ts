import fs from "fs";
import { QueueingSystem, complexityToTime } from "./queueing-system";
import { createTask } from "./task";

const INTERVAL_TIME = 50;
const TOTAL_WORKING_TIME = 3000;
const WORKERS_AMOUNT = 4;
const COMPLETED_TASKS_PATH = "./results.json";
export const TASK_EXECUTION_TIME = complexityToTime.low;

const qs = new QueueingSystem(WORKERS_AMOUNT);
const countsByPeriod: Record<number, number> = {};

let tick = 0;

const intervalId = setInterval(() => {
  qs.addTask(createTask());

  // сбор данных для определения максимальной длины очереди
  countsByPeriod[tick] = qs.taskQueue.queue.length;
  tick++;
}, INTERVAL_TIME);

// Окончание и сбор статистики
setTimeout(() => {
  console.log("completed tasks length: ", qs.completedTasks.length);
  clearInterval(intervalId);
  qs.stop();

  const data = qs.completedTasks.reduce(
    (acc, task, i) => {
      acc.totalTasksTimeInSystem +=
        task.finishedAt.getTime() - task.createdAt.getTime();
      acc.totalTasksTimeInQueue +=
        task.startedToExecute.getTime() - task.createdAt.getTime();
      acc.totalTasksProcessingTime +=
        task.finishedAt.getTime() - task.startedToExecute.getTime();
      return acc;
    },
    {
      totalTasksTimeInSystem: 0,
      totalTasksTimeInQueue: 0,
      totalTasksProcessingTime: 0,
    }
  );

  const maxQueueLength = Math.max(...Object.values(countsByPeriod));
  const averageQueueLength =
    Object.values(countsByPeriod).reduce((acc, val) => acc + val, 0) /
    Object.keys(countsByPeriod).length;

  const resultInfo = {
    totalTime: TOTAL_WORKING_TIME,
    averageTasksTimeInSystem:
      data.totalTasksTimeInSystem / qs.completedTasks.length,
    averageTasksTimeInQueue:
      data.totalTasksTimeInQueue / qs.completedTasks.length,
    averageWorkerIdleTime:
      TOTAL_WORKING_TIME - data.totalTasksProcessingTime / WORKERS_AMOUNT,
    averageQueueLength,
    maxQueueLength,
  };

  fs.writeFileSync(COMPLETED_TASKS_PATH, JSON.stringify(resultInfo));
}, TOTAL_WORKING_TIME);
