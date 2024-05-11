import { TaskProcessor } from "./queueing-system";
import { Task } from "./task";

export class Worker {
  private taskProcessor: TaskProcessor;
  private isBusy: boolean;
  public readonly name: string;

  constructor(name: string, taskProcessor: TaskProcessor) {
    this.name = name;
    this.taskProcessor = taskProcessor;
    this.isBusy = false;
  }

  async processTask(task: Task) {
    if (!this.isBusy) {
      this.isBusy = true;
      await this.taskProcessor(task, this);
      this.isBusy = false;
    }
  }

  isAvailable(): boolean {
    return !this.isBusy;
  }
}
