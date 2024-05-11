import { TaskProcessor } from "./queueing-system";
import { Task } from "./task";
import { numToUpperCaseAscii } from "./utils";
import { Worker } from "./worker";

export class WorkerPool {
  private workers: Worker[];
  private taskProcessor: TaskProcessor;

  constructor(taskProcessor: TaskProcessor, poolSize: number = 4) {
    this.workers = [];
    this.taskProcessor = taskProcessor;

    // Создаем указанное количество исполнителей
    for (let i = 0; i < poolSize; i++) {
      const worker = new Worker(numToUpperCaseAscii(i), this.taskProcessor);
      this.workers.push(worker);
    }
  }

  assignTask(task: Task) {
    const availableWorker = this.workers.find((worker) => worker.isAvailable());
    if (availableWorker) {
      availableWorker.processTask(task);
    } else {
      console.log("No available workers");
    }
  }

  isFull(): boolean {
    return this.workers.every((worker) => !worker.isAvailable());
  }
}
