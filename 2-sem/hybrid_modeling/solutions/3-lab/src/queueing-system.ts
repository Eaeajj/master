import { Task, TaskComplexity } from "./task";
import { PriorityQueue, Comparator } from "./priority-queue";
import { WorkerPool } from "./worker-pool";
import { Worker } from "./worker";
import { fakeExecution } from "./utils";

export const complexityToTime: Record<TaskComplexity, number> = {
  high: 1000,
  medium: 500,
  low: 300,
};

export type TaskProcessor = (task: Task, worker: Worker) => Promise<void>;

export class QueueingSystem {
  readonly taskQueue: PriorityQueue<Task>;
  private workerPool: WorkerPool;
  readonly completedTasks: Task[] = [];
  readonly droppedTasks: Task[] = [];
  private stopped: boolean = false;

  constructor(poolSize: number = 4) {
    // Определяем компаратор для очереди задач
    const taskComparators: Comparator<Task>[] = [
      {
        createdAt: "asc",
      },
      {
        priority: "desc",
      },
    ];

    // Создаем очередь задач
    this.taskQueue = new PriorityQueue<Task>(taskComparators);

    // Создаем пул исполнителей
    this.workerPool = new WorkerPool(
      this.processTask.bind(this) as TaskProcessor,
      poolSize
    );
  }

  // Метод для добавления задачи в очередь
  addTask(task: Task) {
    this.taskQueue.enqueue(task);
    this.processTasks();
  }

  // Метод для обработки задач в очереди
  private processTasks() {
    while (this.taskQueue.size() > 0 && !this.workerPool.isFull()) {
      const task = this.taskQueue.pop();
      if (task) {
        this.workerPool.assignTask(task);
      }
    }
  }

  private async processTask(task: Task, worker: Worker) {
    task.status = "in-progress";
    task.startedToExecute = new Date();

    await fakeExecution(complexityToTime[task.complexity]);
    task.completedByWorker = worker.name;
    task.finishedAt = new Date();
    task.status = "completed";

    if (this.stopped) {
      this.droppedTasks.push(task);
      return;
    }
    this.completedTasks.push(task);

    this.processTasks();
  }

  stop() {
    this.stopped = true;
    this.droppedTasks.push(...this.taskQueue.queue);
    this.taskQueue.queue = [];
  }
}
