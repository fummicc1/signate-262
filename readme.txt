1. �A�m�e�[�V�����̌`��
�E�t�@�C���`��: json�`��
�E�����R�[�h: utf-8
�E�t�@�C����: ***.json (*** = �Ή�����摜�t�@�C�����Ɠ���(***.jpg��***).)
�E�ڍ�:
  - attributes:
      - �N��: "�ÓT��|�ߑ�"
      - �f�[�^���J: "��|�s��"
      - URL
      - �^�C�g��
      - �V���[�Y
      - �ŕ\��
      - ����
      - ����
      - �o�Ŏ�
      - �����i�y�[�W���E�傫���j
      - ISSN
      - ISBN
      - �o�ŔN
      - ���J�͈�
      - ����
      - �ÓT�Ў������
  - labels []:
      - category: "1_overall|2_handwritten|3_typography|4_illustration|5_stamp|6_headline|7_caption|8_textline|9_table"
      - box2d:
          - x1: int
          - y1: int
          - x2: int
          - y2; int
�E���ӓ_:
   - "�N��"�ɂ�����, "�ÓT��"��"�ÓT�Ў���", "�ߑ�"��"�������ȍ~���s����"���Ӗ�����.
   - "�f�[�^���J"�ɂ�����, "��"��"�Ή�����摜����ʌ��J���Ă��悢", "�s��"��"�Ή�����摜����ʌ��J���Ă͂Ȃ�Ȃ�"
     ���Ƃ��Ӗ�����. 
   - (x1, y1, x2, y2)��(left, top, right, bottom)�ɑΉ�����.

2. ��o�t�@�C���̌`��
�E�t�@�C���`��: json�`��
�E�t�@�C����: ***.json (*** = �����̍D���Ȗ��O(e.g. submit))
�E�ڍ�:
  - image_file_0:
      - category_1: [[x1, y1, x2, y2],...]
      - category_2: [[x1, y1, x2, y2],...]
      ...
  - image_file_1:
      - category_1: [[x1, y1, x2, y2],...]
      - category_2: [[x1, y1, x2, y2],...]
      ...
  ...
�E���ӓ_:
  - ���ꂼ��̉摜�ɑ΂���, ���ꂼ��̃��x���ɂ���, bounding box���m�M�x���������ɕ��ׂ邱��.
  - "9_table"�͕]���ΏۊO�ƂȂ�̂�, �\������ۂ͂͂�������.
  - (x1, y1, x2, y2)��(left, top, right, bottom)�ɑΉ�����.
  - ���ꂼ��̉摜�ɑ΂���, ���x���̎�ސ��͈قȂ�.
      - e.g.) image_file_0: ["1_overall", "8_textline"], image_file_1: ["1_overall"]
  - "sample_submit.json"���Q�Ƃ��邱��.