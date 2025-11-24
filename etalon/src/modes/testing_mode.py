from src.utils.logger import get_logger
from src.core.normalize import normalize_all_graphs
from src.utils.file_utils import initialize_register
from src.pipelines.testing_igleng_pipeline import test_model

logger = get_logger(__name__)


def run_testing_mode(args):
    """
    Логіка запуску тестування графів на вплив довжини instance graph на accuracy.

    :param args: Аргументи командного рядка для визначення сценарію навчання.
    """

    try:

        # Завантаження конфігурації
        model_type = args.model_type
        anomaly_type = args.anomaly_type
        action = args.action
        pr_mode = args.pr_mode
        checkpoint_path = args.checkpoint  # наприклад: GATConv_pr_bpmn_seed123_best
        seed = args.quant  # використовуємо наявний int-параметр як seed для узгодженості CLI

        data_file = args.data_file or f"data_{model_type}_{pr_mode}"

        logger.info(f"Запуск режиму навчання для моделі {model_type} з аномалією {anomaly_type}.")

        if action == "test_ig_length":
            # Почати тестування в залежності від довжини instance graph з початку
            logger.info(f"Розпочинається тестування в залежності від довжини instance graph для моделі {model_type}.")
            test_model(
                model_type=model_type,
                anomaly_type=anomaly_type,
                resume=False,
                checkpoint=checkpoint_path or '',
                data_file=data_file,
                pr_mode=pr_mode,
                seed=9467
            )

    except Exception as e:
        logger.error(f"Помилка у режимі навчання: {e}")
        raise